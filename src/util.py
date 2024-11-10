"""Utility functions for processing song data and handling tokenization."""

import datetime
import pickle
import re
import pandas as pd
import tensorflow as tf
from better_profanity import profanity
from . import config

# Compile regex patterns for text processing
star_removal_pattern = re.compile(r"\*")
abbreviated_ing_pattern = re.compile(r"(\win)'(?!\w)(\s)?")

# Common word pairs to simplify text by reducing vocabulary size
word_replacement_pairs = [
    (re.compile(r"(?<!\w)he is(?!\w)"), "he's"),
    (re.compile(r"she is(?!\w)"), "she's"),
    ("cannot", "can't"),
    ("they are", "they're"),
    ("we are", "we're"),
    ("i am", "i'm"),
    ("you are", "you're"),
]

def save_tokenizer(tokenizer, output_path):
    """Save a Keras tokenizer to a specified file path."""
    with open(f"{output_path}/tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(tokenizer_path):
    """Load a Keras tokenizer from a specified file path."""
    with open(tokenizer_path, "rb") as f:
        return pickle.load(f)

def load_song_data(songdata_file, artists=None):
    """Load and filter song data from CSV based on specified artists."""
    print(f"Loading song data from {songdata_file}")
    song_data = pd.read_csv(songdata_file)
    song_data = song_data[~song_data.lyrics.isnull()]
    if artists:
        song_data = song_data[song_data.artist.isin(artists)]
    return song_data.lyrics.values

def load_song_data_chunk(chunk, artists=None, chunk_num=None):
    """Process a chunk of song data with optional filtering by artists."""
    print(f"Processing song data chunk #{chunk_num}")
    song_data_chunk = chunk[~chunk.lyrics.isnull()]
    if artists:
        song_data_chunk = song_data_chunk[song_data_chunk.artist.isin(artists)]
    return song_data_chunk.lyrics.values

def remove_repeated_phrases(song, max_repeats):
    """Reduce repeated phrases within a song to the specified max_repeats limit."""
    lines = song.split("\n")
    unique_lines = []
    line_index = 0

    while line_index < len(lines):
        current_line = lines[line_index].strip()
        unique_lines.append(current_line)
        line_index += 1
        repeat_count = 0

        while line_index < len(lines):
            next_line = lines[line_index].strip()
            if current_line != next_line:
                break
            repeat_count += 1
            if repeat_count < max_repeats:
                unique_lines.append(next_line)
            line_index += 1

    return " \n ".join(unique_lines)

def clean_song_text(song, transform_words, max_repeats, censor_profanity=False):
    """Clean and preprocess a song's lyrics by removing unwanted characters, 
    limiting repetitions, and optionally censoring profanity.
    """
    song = star_removal_pattern.sub("", song)
    song = song.lower().strip("\n")
    song = remove_repeated_phrases(song, max_repeats)
    if censor_profanity:
        song = profanity.censor(song)
    if transform_words:
        song = abbreviated_ing_pattern.sub(r"\1g\2", song)
        for pattern, replacement in word_replacement_pairs:
            song = re.sub(pattern, replacement, song)
    return song

def prepare_song_data(songs, transform_words=False, max_repeats=config.MAX_REPEATS, censor_profanity=False):
    """Prepare a list of songs by applying cleaning and transformation functions."""
    print("Preprocessing song data with proper newline handling.")
    if transform_words:
        print("...also applying word transformations")
    if censor_profanity:
        print("...also censoring profanity")
    start_time = datetime.datetime.now()
    songs = [clean_song_text(song, transform_words, max_repeats, censor_profanity) for song in songs]
    print(f"Preprocessing completed in {datetime.datetime.now() - start_time}")
    return songs

def prepare_tokenizer(songs, max_words=config.MAX_NUM_WORDS, use_char_level=False):
    """Initialize and fit a tokenizer for the given songs."""
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words + 1, char_level=use_char_level)
    tokenizer.filters = tokenizer.filters.replace("\n", "").replace("*", "")
    print("Fitting tokenizer to text data.")
    start_time = datetime.datetime.now()
    tokenizer.fit_on_texts(songs)
    print(f"Tokenizer fitted in {datetime.datetime.now() - start_time}")
    return tokenizer
