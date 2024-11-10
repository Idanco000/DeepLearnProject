"""Train a song generation model using lyrics data."""

import argparse
import datetime
import os
import statistics

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau


from . import config, embedding, util


class LyricsDataGenerator(tf.keras.utils.Sequence):
    """Data generator for loading and batching large song datasets."""

    def __init__(self, songdata_file, tokenizer, batch_size=32, seq_length=27, chunksize=5000):
        self.songdata_file = songdata_file
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.chunksize = chunksize
        self.chunk_iter = pd.read_csv(songdata_file, chunksize=self.chunksize)
        self.chunk = None
        self.chunk_index = 0

    def __len__(self):
        """Calculate the number of batches in the generator."""
        return int(np.floor(len(self.chunk) / self.batch_size)) if self.chunk is not None else 0

    def __getitem__(self, index):
        """Retrieve a batch of data at the specified index."""
        if self.chunk is None or self.chunk_index >= len(self.chunk):
            try:
                self.chunk = next(self.chunk_iter)
                self.chunk_index = 0
            except StopIteration:
                self.chunk_iter = pd.read_csv(self.songdata_file, chunksize=self.chunksize)
                self.chunk = next(self.chunk_iter)
                self.chunk_index = 0

        batch_data = self.chunk.iloc[self.chunk_index:self.chunk_index + self.batch_size]
        self.chunk_index += self.batch_size

        X_batch, y_batch = self._process_data(batch_data['lyrics'])
        return np.array(X_batch), np.array(y_batch)

    def _process_data(self, lyrics_batch):
        """Process lyrics data into input-output pairs for training."""
        X_batch, y_batch = [], []
        for lyrics in lyrics_batch:
            encoded = self.tokenizer.texts_to_sequences([lyrics])[0]
            for i in range(1, len(encoded)):
                seq = encoded[:i]
                if len(seq) < self.seq_length:
                    seq = [0] * (self.seq_length - len(seq)) + seq
                else:
                    seq = seq[-self.seq_length:]
                X_batch.append(seq)
                y_batch.append(encoded[i])
        return X_batch, y_batch


def prepare_data(
    songs,
    transform_words=False,
    use_full_sentences=False,
    use_strings=False,
    num_lines_to_include=config.NUM_LINES_TO_INCLUDE,
    max_repeats=config.MAX_REPEATS,
    char_level=False,
    profanity_censor=False,
    max_num_words=config.MAX_NUM_WORDS,
):
    """Prepare songs for training, including tokenizing and word preprocessing.

    Parameters
    ----------
    songs : list
        A list of song strings
    transform_words : bool
        Whether or not to transform certain words such as cannot -> can't
    use_full_sentences : bool
        Whether or not to only create full sentences, i.e. sentences where
        all the tokenized words are non-zero.
    use_strings : bool
        Whether or not to return sequences as normal strings or lists of integers
    num_lines_to_include: int
        The number of lines to include in the sequences. A "line" is found by
        taking the median length of lines over all songs.
    max_repeats: int
        The number of times a sentence can repeat between newlines
    char_level: bool
        Whether or not to prepare for character-level modeling or not. The
        default is False, meaning the data is prepared to word-level

    Returns
    -------
    X : list
        Input sentences
    y : list
        Predicted words
    seq_length : int
        The length of each sequence
    num_words : int
        Number of words in the vocabulary
    tokenizer : object
        The Keras preproceessing tokenizer used for transforming sentences.

    """
    songs = util.prepare_songs(
        songs,
        transform_words=transform_words,
        max_repeats=max_repeats,
        profanity_censor=profanity_censor,
    )
    tokenizer = util.prepare_tokenizer(
        songs, char_level=char_level, num_words=max_num_words
    )

    num_words = min(max_num_words, len(tokenizer.word_index))

    print("Encoding all songs to integer sequences")
    if use_full_sentences:
        print("Note: Will only use full integer sequences!")
    now = datetime.datetime.now()
    songs_encoded = tokenizer.texts_to_sequences(songs)
    print("Took {}".format(datetime.datetime.now() - now))
    print()

    newline_int = tokenizer.word_index.get("\n", None)
    if newline_int is None:
        raise ValueError("Newline character '\\n' not found in tokenizer vocabulary.")

    # Calculate the average/median length of each sentence before a newline is seen.
    line_lengths = []
    print("Find the average/median line length for all songs")
    now = datetime.datetime.now()
    for song_encoded in songs_encoded:
        # Find the indices of the newline characters.
        # For double newlines (between verses), the distance will be 1 so these
        # distances are ignored...

        # Note: np.where() returns indices when used only with a condition.
        # Thus, these indices can be used to measure the distance between
        # newlines.
        newline_indexes = np.where(np.array(song_encoded) == newline_int)[0]

        if len(newline_indexes) == 0:
            continue  # Skip if there are no newline characters in the song

        lengths = [
            newline_indexes[i] - newline_indexes[i - 1] - 1
            for i in range(1, len(newline_indexes))
            if newline_indexes[i] - newline_indexes[i - 1] > 1
        ]
        line_lengths.extend(lengths)

        line_lengths.append(newline_indexes[0])
        line_lengths.append(len(song_encoded) - newline_indexes[-1] - 1)

    print("Took {}".format(datetime.datetime.now() - now))
    print()

    median_seq_length = statistics.median(line_lengths)
    mean_seq_length = statistics.mean(line_lengths)
    print(
        "Median/mean line length from {} lines: {}/{}".format(
            len(line_lengths), median_seq_length, mean_seq_length
        )
    )
    print(f"Will include {num_lines_to_include} lines for sequences.")
    print()

    seq_length = (
        int(round(median_seq_length)) * num_lines_to_include + num_lines_to_include - 1
    )

    X, y = [], []
    print("Creating training data")
    now = datetime.datetime.now()
    for song_encoded in songs_encoded:
        start_index = seq_length if use_full_sentences else 1
        for i in range(start_index, len(song_encoded)):
            seq = song_encoded[:i]
            if len(seq) < seq_length:
                seq.extend([0] * (seq_length - len(seq)))
            seq = seq[-seq_length:]
            X.append(seq)
            y.append(song_encoded[i])
    print("Took {}".format(datetime.datetime.now() - now))
    print()

    if use_strings:
        X = tokenizer.sequences_to_texts(X)

    print(f"Total number of samples: {len(X)}")

    return X, y, seq_length, num_words, tokenizer

def should_use_generator(songdata_file, use_generator_flag):
    """Data generator option checker"""
    if use_generator_flag:
        return True
    file_size = os.path.getsize(songdata_file)
    return file_size > LARGE_FILE_THRESHOLD


def create_model(
    seq_length,
    num_words,
    embedding_matrix,
    embedding_dim=config.EMBEDDING_DIM,
    embedding_not_trainable=False,
    tfjs_compatible=False,
    gpu_speedup=False,
):
    """Creating an empty model befor training"""
    if not tfjs_compatible:
        print("Model will be created without tfjs support")

    if gpu_speedup:
        print("Model will be created with better GPU compatibility")

    # The + 1 accounts for the OOV token
    actual_num_words = num_words + 1

    inp = tf.keras.layers.Input(shape=(seq_length,))
    x = tf.keras.layers.Embedding(
        input_dim=actual_num_words,
        output_dim=embedding_dim,
        input_length=seq_length,
        weights=[embedding_matrix] if embedding_matrix is not None else None,
        mask_zero=True,
        name="song_embedding",
    )(inp)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            256, return_sequences=True, reset_after=gpu_speedup or not tfjs_compatible  # Increased units to 256
        )
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # Added Dropout
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            256,  # Increased units to 256
            dropout=0.2,
            recurrent_dropout=0.0 if gpu_speedup else 0.2,
            reset_after=gpu_speedup or not tfjs_compatible,
        )
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # Added Dropout
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outp = tf.keras.layers.Dense(actual_num_words, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=[inp], outputs=[outp])

    if embedding_not_trainable:
        model.get_layer("song_embedding").trainable = False

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"],
    )
    model.summary()
    return model

def create_transformer_model(
    num_words,
    transformer_network,
    trainable=True,
):
    """Use a transformer architecture like the universal sentence encoder or BERT"""
    inp = tf.keras.layers.Input(shape=[], dtype=tf.string)

    if transformer_network == "use":
        x = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder/4",
            trainable=trainable,
            input_shape=[],
            dtype=tf.string,
        )(inp)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
    elif transformer_network == "bert":
        # XXX: This is the smallest possible bert encoder. We can't expect wonders.
        x = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")(
            inp
        )
        outputs = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2",
            trainable=trainable,
        )(x)
        x = outputs["pooled_output"]
        x = tf.keras.layers.Dropout(0.1)(x)

    # The + 1 accounts for the OOV token which can sometimes be present as the target word
    outp = tf.keras.layers.Dense(num_words + 1, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=inp, outputs=outp)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def train(
    export_dir=None,
    songdata_file=config.SONGDATA_FILE,
    artists=config.ARTISTS,
    embedding_file=config.EMBEDDING_FILE,
    embedding_dim=config.EMBEDDING_DIM,
    embedding_not_trainable=False,
    transform_words=False,
    use_full_sentences=False,
    transformer_network=None,
    num_lines_to_include=config.NUM_LINES_TO_INCLUDE,
    batch_size=config.BATCH_SIZE,
    max_epochs=config.MAX_EPOCHS,
    tfjs_compatible=False,
    gpu_speedup=False,
    save_freq=config.SAVE_FREQUENCY,
    max_repeats=config.MAX_REPEATS,
    char_level=False,
    early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
    profanity_censor=False,
    max_num_words=config.MAX_NUM_WORDS,
    use_generator=False,
):
    """Training the generative model based on our dataset.
        If dataset is too big we will train it with LyricsDataGenerator which will generate chunks for training"""
    
    if export_dir is None:
        export_dir = "./export/{}".format(
            datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")
        )
        os.makedirs(export_dir, exist_ok=True)
    
    if use_generator:
        print("Using generator for large dataset.")
        tokenizer = util.prepare_tokenizer(pd.read_csv(songdata_file, usecols=['lyrics'])['lyrics'])
        embedding_matrix = embedding.create_embedding_matrix(
            tokenizer,
            embedding.create_embedding_mappings(embedding_file=embedding_file),
            embedding_dim=embedding_dim,
            max_num_words=max_num_words
        )
        
        model = create_model(
            seq_length=num_lines_to_include,
            num_words=max_num_words,
            embedding_matrix=embedding_matrix,
            embedding_dim=embedding_dim,
            embedding_not_trainable=embedding_not_trainable,
            tfjs_compatible=tfjs_compatible,
            gpu_speedup=gpu_speedup,
        )

        training_generator = LyricsDataGenerator(songdata_file, tokenizer, batch_size=batch_size)
        model.fit(
            training_generator,
            epochs=max_epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    patience=early_stopping_patience,
                    verbose=1,
                    min_delta=0.001,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    "{}/model.h5".format(export_dir),
                    monitor="loss",
                    save_best_only=True,
                    save_freq=save_freq,
                    verbose=1,
                ),
                ReduceLROnPlateau(
                    monitor='loss',
                    factor=0.5,
                    patience=2,
                    verbose=1,
                    min_lr=1e-5
                ),
            ],
        )
    else:
        songs = util.load_songdata(songdata_file=songdata_file, artists=artists)
        print(
            f"Will use {len(songs)} songs from {len(artists) if len(artists) > 0 else 'all'} artists"
        )

        X, y, seq_length, num_words, tokenizer = prepare_data(
            songs,
            transform_words=transform_words,
            use_full_sentences=use_full_sentences,
            use_strings=bool(transformer_network),
            num_lines_to_include=num_lines_to_include,
            max_repeats=max_repeats,
            char_level=char_level,
            profanity_censor=profanity_censor,
            max_num_words=max_num_words,
        )
        util.pickle_tokenizer(tokenizer, export_dir)

        model = None

        if transformer_network:
            print(f"Using transformer network '{transformer_network}'")
            model = create_transformer_model(
                num_words, transformer_network, trainable=not embedding_not_trainable
            )
            # Some transformer networks are slow to save, let's just save it every epoch.
            save_freq = "epoch" if transformer_network == "use" else save_freq
        else:
            embedding_matrix = None
            # Don't use word embeddings on char-level training.
            if not char_level:
                print(f"Using precreated embeddings from {embedding_file}")
                embedding_mapping = embedding.create_embedding_mappings(
                    embedding_file=embedding_file
                )
                embedding_matrix = embedding.create_embedding_matrix(
                    tokenizer,
                    embedding_mapping,
                    embedding_dim=embedding_dim,
                    max_num_words=num_words,
                )
            model = create_model(
                seq_length,
                num_words,
                embedding_matrix,
                embedding_dim=embedding_dim,
                embedding_not_trainable=embedding_not_trainable,
                tfjs_compatible=tfjs_compatible,
                gpu_speedup=gpu_speedup,
            )
        # Define the learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(
            monitor='loss',       # Metric to monitor
            factor=0.5,           # Factor by which to reduce the learning rate
            patience=2,           # Number of epochs with no improvement before reducing the rate
            verbose=1,            # Print update messages
            min_lr=1e-5           # Minimum learning rate
        )

        print(
            f"Running training with batch size {batch_size} and maximum epochs {max_epochs}"
        )


        # Run the training
        model.fit(
            np.array(X),
            np.array(y),
            batch_size=batch_size,
            epochs=max_epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    patience=early_stopping_patience,
                    verbose=1,
                    min_delta=0.001,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    "{}/model.h5".format(export_dir),
                    monitor="loss",
                    save_best_only=True,
                    save_freq=save_freq,
                    verbose=1,
                ),
                lr_scheduler  # Add the learning rate scheduler here
            ],
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artists", default=config.ARTISTS, help=config.help_artists, nargs="*")
    parser.add_argument("--songdata-file", default=config.SONGDATA_FILE, help="Use a custom songdata file")
    parser.add_argument("--embedding-file", default=config.EMBEDDING_FILE, help="Use a custom embedding file")
    parser.add_argument("--embedding-not-trainable", action="store_true", help=config.help_embedding_nt)
    parser.add_argument("--transform-words", action="store_true",  help=config.help_transform_word)
    parser.add_argument("--use-full-sentences", action="store_true", help=config.help_full_sentences)
    parser.add_argument("--transformer-network", help=config.help_transformer, choices=["use", "bert"])
    parser.add_argument("--num-lines-to-include", type=int, default=config.NUM_LINES_TO_INCLUDE, help=config.help_line_num    )
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for training",)
    parser.add_argument("--max-epochs", type=int, default=config.MAX_EPOCHS, help="Maximum number of epochs to train for",)
    parser.add_argument("--tfjs-compatible", action="store_true", help=config.help_js)
    parser.add_argument("--gpu-speedup", action="store_true", help=config.help_GRU)
    parser.add_argument("--max-repeats", type=int, default=config.MAX_REPEATS, help=config.help_max_rep)
    parser.add_argument("--save-freq", type=int, default=config.SAVE_FREQUENCY, help=config.help_save_freq)
    parser.add_argument("--char-level", action="store_true", help=config.help_char_lev)
    parser.add_argument("--early-stopping-patience", type=int, default=config.EARLY_STOPPING_PATIENCE, help=config.help_patience)
    parser.add_argument("--profanity-censor", action="store_true", help=config.help_profanity)
    parser.add_argument("--max-num-words", type=int, default=config.MAX_NUM_WORDS, help=config.help_max_words)
    parser.add_argument("--embedding", default="word2vec", choices=["word2vec", "use"], help="Choose embedding model: 'word2vec' or 'use'")
    parser.add_argument("--use-generator", action="store_true", help="Force use of data generator for training")
    
    args = parser.parse_args()
    artists = args.artists if args.artists != ["*"] else []
    train(
        songdata_file=args.songdata_file,
        artists=artists,
        embedding_file=args.embedding_file,
        transform_words=args.transform_words,
        use_full_sentences=args.use_full_sentences,
        embedding_not_trainable=args.embedding_not_trainable,
        transformer_network=args.transformer_network,
        num_lines_to_include=args.num_lines_to_include,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        tfjs_compatible=args.tfjs_compatible,
        gpu_speedup=args.gpu_speedup,
        max_repeats=args.max_repeats,
        save_freq=args.save_freq,
        char_level=args.char_level,
        early_stopping_patience=args.early_stopping_patience,
        profanity_censor=args.profanity_censor,
        max_num_words=args.max_num_words,
    )
    
    

# Threshold for large file size in bytes (e.g., 2 GB)
LARGE_FILE_THRESHOLD = 2 * 1024 * 1024 * 1024


