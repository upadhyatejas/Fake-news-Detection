import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from fake_news_detection.load_news import train_data, train_labels

MAX_NB_WORDS = 50000            # dictionary size
MAX_SEQUENCE_LENGTH = 1500      # max word length of each individual article
EMBEDDING_DIM = 300             # dimensionality of the embedding vector (50, 100, 200, 300)
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')


def tokenize_trainingdata(texts, labels):
    tokenizer.fit_on_texts(texts)
    pickle.dump(tokenizer, open('Models/tokenizer.p', 'wb'))

    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences=sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(labels, num_classes=len(set(labels)))

    return data, labels, word_index


X, Y, word_index = tokenize_trainingdata(train_data, train_labels)

# split the data (90% train, 5% test, 5% validation)
train_data = X[:int(len(X)*0.9)]
train_labels = Y[:int(len(X)*0.9)]
test_data = X[int(len(X)*0.9):int(len(X)*0.95)]
test_labels = Y[int(len(X)*0.9):int(len(X)*0.95)]
valid_data = X[int(len(X)*0.95):]
valid_labels = Y[int(len(X)*0.95):]
