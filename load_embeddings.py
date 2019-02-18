from keras.layers import Embedding

from fake_news_detection.tokenize_training import EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, word_index
import numpy as np


def load_embeddings(word_index, embeddingsfile='news_dataset/wordEmbeddings/glove.6B.%id.txt' % EMBEDDING_DIM):
    embeddings_index = {}
    f = open(embeddingsfile, 'r', encoding='utf8')
    for line in f:
        values = line.split(' ')        # split line by spaces
        word = values[0]                # each line starts with the word
        coefs = np.asarray(values[1:], dtype='float32')     # the rest of the line is the vector
        embeddings_index[word] = coefs      # put into embedding dictionary
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embeddings_matrix = np.zeros(len(word_index) + 1, EMBEDDING_DIM)
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embeddings_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    return embedding_layer


embedding_layer = load_embeddings(word_index=word_index)
