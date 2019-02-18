from keras import Sequential, Model, Input
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from keras import callbacks

from fake_news_detection.load_embeddings import embedding_layer
from fake_news_detection.tokenize_training import MAX_SEQUENCE_LENGTH, train_data, train_labels, valid_data, \
    valid_labels


def baseline_model(sequence_input, embedded_sequences, classes=2):
    x = Conv1D(64, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(256, 2, activation='relu')(x)
    x = GlobalAveragePooling1D(5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(classes, activation='softmax')(x)

    model = Model(sequence_input, preds)
    return model


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

model = baseline_model(sequence_input, embedded_sequences, classes=2)

model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=['acc'])

print(model.summary())

model.fit(train_data, train_labels,
          validation_data=(valid_data, valid_labels),
          epochs=25, batch_size=64)
