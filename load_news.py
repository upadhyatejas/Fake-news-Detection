import pandas as pd
import numpy as np
import random


def load_news():
    df = pd.read_csv('fake-news/train.csv', encoding='utf8')
    train_data = df['text'].values          # 'text' column contains articles
    train_labels = df['label'].values      # 'label' column contains label

    # Randomly shuffle data and labels together
    zipped = list(zip(train_data, train_labels))
    random.shuffle(zipped)
    train_data, train_labels = zip(*zipped)
    del df      # clear the memory

    return np.array(train_data), np.array(train_labels)

print("i have reached before the end of the function" )
train_data, train_labels = load_news()
print("i have reached the end")