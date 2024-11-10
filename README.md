# Song Lyrics Generator

## Project Overview
In this project we train a rnn model which is based on layers from keras, we tried many different hyperparameters and and models (also transformers), and got to the conclusion that the best way is to train an embedding model based on data we collected online for generating songs lyrics.

### Objective
The goal of this project is to generate high quality songs using two approaches:
1. Fine-tuned **RNN** model with keras.
2. Fine-tuned **Transformer** model using BERT or USE.

## Motivation
We both like music, and after using SUNO application we wanted to try and generate our own songs based of a model that we train.


## Dataset
The dataset used is the [Genius Song Lyrics](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information) from Kaggle.
It contains data of  song like Genre,artist,year,views... but we used only the lyrics,	furthermore most entries are songs, but there are also some books, poems and even some other stuff

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <repository_url>
cd DeepLyricsProject

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


## Model Architectures

### Neural network architecture (e.g., LSTM, 	Transformer, RNN)
Key Components:
	 Embedding layers, recurrent layers

## Methods and Training
Both models were fine-tuned with the following process:
- Tokenized input fed into the model.
- **Loss Function**: sparse_categorical_crossentropy.
- **Optimizer**: rmsprop,
- **Early Stopping and DropOut**: Implemented to prevent overfitting.

## Experiments and Results
The experiments focused on evaluating model performance based on our personal preferences after training models with different embedding dimensions, dropout, LR Scheduler and sizes of datasets

| Model   | Accuracy | Precision | Recall | F1-score |
|---------|----------|-----------|--------|----------|
| GPT-2   | 79.4%    | 73.52%    | 74.27% | 73.89%   |
| BERT    | 78.75%   | 80.72%    | 85.43% | 83.00%   |
| GPT-4o  | 60.9%    | -         | -      | -        |

Observations:
- The **GPT-2 model** performed well in next-token prediction tasks, showing good capture of patterns in the "Gaming and Entertainment" category.
- **LoRa** proved to be an effective technique, reducing time and memory usage while achieving high performance.

## Future Work
There are several potential extensions for this project:
- Expanding the model to support **music generation**.
- Adding more parameters to the learning so it will take into account the year of publish, generes, artist and more
- Training diffrent models based on different languages, and see if the model architecture give different qualities for different languages

