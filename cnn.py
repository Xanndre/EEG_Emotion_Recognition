import pandas as pd
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import train_test_split
import numpy as np
import helpers
import tensorflow as tf

NUM_OF_LABELS = 3
NUM_OF_EPOCHS = 100
BATCH_SIZE = 16


def create_model(n_timesteps, n_features):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))

    model.add(Dense(NUM_OF_LABELS, activation='softmax'))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam', metrics=['accuracy'])

    return model


def evaluate_model(model, x_train, y_train, x_test, y_test):
    history = model.fit(x_train, y_train, epochs=NUM_OF_EPOCHS,
                        batch_size=BATCH_SIZE, verbose=2, validation_data=(x_test, y_test))

    model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=2)

    helpers.draw_plot(history)


def main():
    data = pd.read_csv("features.csv")

    x = data.iloc[:, 0:216].to_numpy()
    y = data.iloc[:, -1:].to_numpy()

    x = x.reshape((x.shape[0], x.shape[1], 1))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=100)

    n_timesteps, n_features = x_train.shape[1], x_train.shape[2]

    model = create_model(n_timesteps, n_features)

    evaluate_model(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
