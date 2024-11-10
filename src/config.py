"""Configuration file for setting parameters and model hyperparameters."""

# Data and file parameters
SONGDATA_FILE = "./data/song_lyrics.csv"
EMBEDDING_FILE = "./data/word2vec.txt"

# Training parameters
BATCH_SIZE = 256
MAX_EPOCHS = 100
SAVE_FREQUENCY = 10
EARLY_STOPPING_PATIENCE = 5

# Model parameters
EMBEDDING_DIM = 256
MAX_NUM_WORDS = 20000
MAX_REPEATS = 2
NUM_LINES_TO_INCLUDE = 4

# Artists list
ARTISTS = [
    "The Beatles", "Rolling Stones", "Pink Floyd", "Queen", "Who", "Jimi Hendrix",
    "Doors", "Nirvana", "Eagles", "Aerosmith", "Creedence Clearwater Revival",
    "Guns N' Roses", "Black Sabbath", "U2", "David Bowie", "Beach Boys",
    "Van Halen", "Bob Dylan", "Eric Clapton", "Red Hot Chili Peppers",
]


## Help texts for arguments
help_artists = """
            A list of artists to use. Use '*' (quoted) to include everyone.
            The default is a group of rock artists.
        """
help_embedding_nt = """
            Whether the embedding weights are trainable or locked to the
            vectors of the embedding file. It is only recommend to set this
            flag if the embedding file contains vectors for the full
            vocabulary of the songs.
        """
help_transform_word = """
            To clean the song texts a little bit more than normal by e.g.
            transforming certain words like runnin' to running.
        """     
help_full_sentences = """
            Use only full sentences as training input to the model, i.e. no
            single-word vectors will be used for training. This decreases the
            training data, and avoids putting emphasis on single starting
            words in a song.
        """
help_transformer = """
            Use a transformer architecture like the universal sentence encoder
            rather than a recurrent neural network.
        """
help_line_num = """
            Number of lyrics lines to include. The data preparation finds a
            median and average line length (typically between 5-10 words) and
            includes a number of these standard lines according to this
            parameter. This ensures all sequences are the same length but it
            might chop up some songs mid-sentences.
        """
help_js = """
            Makes the model exportable to JavaScript (Tensorflow JS). When
            enabled, the network structure is changed slightly for the
            recurrent GRU cells so they are supported by Tensorflow JS,
            specifically setting reset_after=False. Note that this will
            disable GPU training, which might (or might not) slow things
            down.

            This flag is ignored when using transformers, since they are not
            compatible in the first place.
        """
help_GRU = """
            Make adjustments to the recurrent unit settings in the network to
            allow using a cuDNN-specific implementation for a potential speedup.
            See https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU
        """
help_max_rep = """
            If a sentences repeats multiple times (for example in a very long
            and repeating chorus), reduce the number of repeats for model
            training to this number. Repeats are delimited by a newline for
            simplicity.
            By default, anything above 2 repeats are discarded for training.
        """
help_save_freq = """
        How often to save a snapshot of the model (if it has improved
        since last snapshot). Model saving can take some time so if batches are
        very fast, you might want to increase this number.
        """
help_char_lev = """
        Determines whether to use a character-level model, i.e. the
        model will predict the next character instead of the next word.
        """       
help_patience = """
        How many epochs with no loss improvements before doing early
        stopping. For small datasets, you might want to increase this.
        """
help_profanity = """
        Replace certain words with **** during preprocessing training.
        This eliminates some of the bad words that artists might use. This can
        be useful for presentations :-)
        """
help_max_words = """
                Maximum number of words to include in the output.
                """
        