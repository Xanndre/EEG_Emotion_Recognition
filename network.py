import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import helpers
import selection
import tensorflow as tf


NUM_OF_LABELS = 3
NUM_OF_EPOCHS = 100
BATCH_SIZE = 32
ANOVA_FEATURES = 20

# possible: band, anova, channels_4, channels_6, channels_11, channels_28, all
FEATURES_TYPE = 'all'
# possible: seed, mahnob
DATASET = 'mahnob'

NUM_OF_FEATURES = 310 if DATASET == 'seed' else 160
FILE_NAME = 'mahnob_features.csv' if DATASET == 'mahnob' else 'seed_features.csv'

BAND_NAME = 'Beta'
LABEL_TYPE = 'Arousal_Mapped'
CHANNELS = helpers.get_channels(FEATURES_TYPE)


def create_model(n_features):
    model = Sequential()
    model.add(Dense(128, activation='relu',
                    input_shape=(n_features,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(NUM_OF_LABELS, activation='softmax'))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam', metrics=['accuracy'])

    return model


def evaluate_model(model, x_train, y_train, x_test, y_test):
    history = model.fit(x_train, y_train, epochs=NUM_OF_EPOCHS,
                        validation_split=0.2,
                        batch_size=BATCH_SIZE, verbose=2)

    model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=2)

    helpers.draw_confusion_matrix(model.predict(x_test), y_test)
    helpers.draw_plot(history)


def main():
    data = pd.read_csv(FILE_NAME)

    if FEATURES_TYPE == 'band':
        x = selection.select_band_features(BAND_NAME, data)

    elif FEATURES_TYPE in ['channels_4', 'channels_6', 'channels_11', 'channels_28']:
        x = selection.select_channel_features(CHANNELS, data)

    else:
        x = data.iloc[:, 0:NUM_OF_FEATURES].to_numpy()

    y = data[LABEL_TYPE].to_numpy()

    if FEATURES_TYPE == 'anova':
        x = selection.select_anova_features(ANOVA_FEATURES, x, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=100, shuffle=True)

    model = create_model(x_train.shape[1])

    evaluate_model(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()