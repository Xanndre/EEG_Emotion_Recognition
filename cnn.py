import pandas as pd
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import train_test_split
import helpers
import selection
import tensorflow as tf

NUM_OF_LABELS = 3
NUM_OF_EPOCHS = 100
BATCH_SIZE = 16
# possible: band, anova, channels_4, channels_6, channels_11, channels_28, all
FEATURES_TYPE = 'channels_28'
BAND_NAME = 'Beta'
LABEL_TYPE = 'Valence'
IS_PAIR = False
CHANNELS_OUT = helpers.get_channel_out_names(FEATURES_TYPE)


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

    if FEATURES_TYPE == 'band':
        x = selection.select_band_features(BAND_NAME, data, IS_PAIR)

    elif FEATURES_TYPE in ['channels_4', 'channels_6', 'channels_11', 'channels_28']:
        x = selection.select_channel_features(CHANNELS_OUT, data, IS_PAIR)

    else:
        x = data.iloc[:, 0:216].to_numpy()

    y = data[LABEL_TYPE].to_numpy()

    x = x.reshape((x.shape[0], x.shape[1], 1))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=100)

    n_timesteps, n_features = x_train.shape[1], x_train.shape[2]

    model = create_model(n_timesteps, n_features)

    evaluate_model(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
