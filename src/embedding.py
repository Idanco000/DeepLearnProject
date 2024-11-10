"""Embedding utilities."""
import argparse
import csv
import datetime
import multiprocessing

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec

from . import config, util

_num_workers = 1
try:
    _num_workers = multiprocessing.cpu_count()
except:
    print("Only one worker")
    pass

def create_word2vec_in_chunks(data_dir="./data", songdata_file=config.SONGDATA_FILE,
    artists=config.ARTISTS,
    embedding_dim=config.EMBEDDING_DIM,
    name_suffix="",
    chunksize=5000  
):
    """Train a Word2Vec model on song lyrics in chunks to handle large datasets.
    
    Parameters:
    - data_dir (str): Directory to save the trained model.
    - file_path (str): Path to the song lyrics file.
    - artists (list): List of artists to filter the song data.
    - embedding_dim (int): Dimension of the word embeddings.
    - name_suffix (str): Suffix for the saved model's file name.
    - chunksize (int): Number of rows per chunk to load and process.
    """
    # Initialize an empty Word2Vec model with given parameters
    model = Word2Vec(
        size=embedding_dim, workers=_num_workers, min_count=1
    )
    is_first_chunk = True

    # Read Excel file in chunks
    print(f"Reading and training on {songdata_file} in chunks.", flush=True)
    chunk_num = 0
    for chunk in pd.read_csv(songdata_file, chunksize=chunksize):
        chunk_num += 1
        # Preprocess each chunk (filter artists if needed, and clean songs)
        chunk_songs = util.load_songdata_chunk(chunk, artists, chunk_num) 
        chunk_songs = util.prepare_songs(chunk_songs)

        # Tokenize songs in the current chunk
        sequences = [
            tf.keras.preprocessing.text.text_to_word_sequence(song) for song in chunk_songs
        ]

        if is_first_chunk:
            # Build vocabulary on the first chunk
            model.build_vocab(sequences, update=False)
            is_first_chunk = False
        else:
            # Update vocabulary for additional chunks
            model.build_vocab(sequences, update=True)

        # Train Word2Vec on the current chunk
        model.train(sequences, total_examples=len(sequences), epochs=model.epochs)
        print("Processed a chunk", flush=True)

    # Save the model after training on all chunks
    model.save(f"{data_dir}/word2vec{name_suffix}.model")
    # Also save the model as a text file
    with open(f"{data_dir}/word2vec{name_suffix}.txt", "w") as f:
        writer = csv.writer(f, delimiter=" ", quoting=csv.QUOTE_MINIMAL)
        for word in model.wv.vocab:
            word_vector = model.wv[word]
            writer.writerow([word] + ["{:.5f}".format(i) for i in word_vector])

    return model


def create_word2vec(data_dir="./data", songdata_file=config.SONGDATA_FILE, artists=config.ARTISTS, embedding_dim=config.EMBEDDING_DIM,
                    name_suffix=""):
    # loading the dataset and cleaning up the songs
    songs = util.load_songdata(songdata_file=songdata_file, artists=artists)
    songs = util.prepare_songs(songs)

    sequences = []
    # Tokenize songs
    for song in songs:
        sequences.append(tf.keras.preprocessing.text.text_to_word_sequence(song))

    # Initializing the model also starts training
    print(
        "Training Word2Vec on {} sequences with {} workers".format(
            len(sequences), _num_workers
        )
    )
    now = datetime.datetime.now()
    model = Word2Vec(sequences, size=embedding_dim, workers=_num_workers, min_count=1)
    print("Took {}".format(datetime.datetime.now() - now))

    # Save the model both in the gensim format but also as a text file, similar
    # to the glove embeddings
    model.save(f"{data_dir}/word2vec{name_suffix}.model")
    with open(f"{data_dir}/word2vec{name_suffix}.txt", "w") as f:
        writer = csv.writer(f, delimiter=" ", quoting=csv.QUOTE_MINIMAL)
        for word in model.wv.vocab:
            word_vector = model.wv[word]
            writer.writerow([word] + ["{:.5f}".format(i) for i in word_vector])

    return model


def create_embedding_mappings(embedding_file=config.EMBEDDING_FILE):
    """Generate a dictionary of word embeddings from a text file.
    
    Parameters:
    - embedding_file (str): Path to the embedding file containing word vectors.
    
    Returns:
    - dict: A dictionary mapping words to their embedding vectors.
    """
    print(f"Loading embedding mapping file {embedding_file}", flush=True)
    embedding = pd.read_table(
        embedding_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE
    )
    mapping = {}

    print("Creating embedding mappings for faster lookup", flush=True)
    for row in embedding.itertuples():
        mapping[row[0]] = list(row[1:])

    return mapping


def create_embedding_matrix(
    tokenizer,
    embedding_mapping,
    embedding_dim=config.EMBEDDING_DIM,
    max_num_words=config.MAX_NUM_WORDS,
):
    """Create an embedding matrix from a tokenizer and an embedding mapping.
    
    Parameters:
    - tokenizer: Tokenizer with word index from the training data.
    - embedding_mapping (dict): Dictionary of pre-trained word embeddings.
    - embedding_dim (int): Dimension of the embedding vectors.
    - max_num_words (int): Maximum number of words to include in the matrix.
    
    Returns:
    - np.array: Embedding matrix for use in a Keras Embedding layer.
    """
    print("Creating embedding matrix", flush=True)

    # Create embedding matrix, add an extra row for the out of vocabulary vector
    embedding_matrix = np.zeros((max_num_words + 1, embedding_dim))

    num_words = 0
    num_words_found = 0
    num_words_ignored = 0

    now = datetime.datetime.now()
    for word, i in tokenizer.word_index.items():
        if i > max_num_words:
            num_words_ignored += 1
            continue
        num_words += 1
        if word in embedding_mapping:
            embedding_matrix[i] = embedding_mapping[word]
            num_words_found += 1
    print("Took {}".format(datetime.datetime.now() - now), flush=True)
    print(
        "Found {} words in mapping ({:.1%})".format(
            num_words_found, num_words_found / num_words
        )
    )
    print("{} words were ignored because they are infrequent".format(num_words_ignored), flush=True)

    return embedding_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-suffix", default="", help="Name suffix for the embedding file.")
    parser.add_argument("--songdata-file", default=config.SONGDATA_FILE, help="Use a custom songdata file")
    parser.add_argument("--artists", default=config.ARTISTS, help=config.help_artists, nargs="*")
    parser.add_argument("--chunksize", type=int, default=8000, help="Number of rows per chunk to read")
    args = parser.parse_args()
    print(args.artists, flush=True)
    if args.artists != ["*"]:
        artists = args.artists
        create_word2vec(name_suffix=args.name_suffix, artists=artists, songdata_file=args.songdata_file)
    else: 
        artists = []
        create_word2vec_in_chunks(
            name_suffix=args.name_suffix, 
            songdata_file=args.songdata_file, 
            artists=artists,
            chunksize=args.chunksize
        )
    


