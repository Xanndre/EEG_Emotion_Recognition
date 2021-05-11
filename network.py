import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import helpers
import selection
import tensorflow as tf
import numpy as np

NUM_OF_LABELS = 4
NUM_OF_EPOCHS = 500
BATCH_SIZE = 32
SELECTED_FEATURES = 25
RANDOM_SEED = 289

NUM_OF_FEATURES = 310
FILE_NAME = 'seed_features_de_movingAve.csv'

# possible: all, band, anova, chi2, channels_4, channels_6, channels_11, channels_28, channels_32,
#           channels_10, channels_9, channels_8, channels_7, channels_37, channels_42, channels_50
FEATURES_TYPE = 'chi2'

BAND_NAME = 'Beta'
LABEL_TYPE = 'Label'
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
                        batch_size=BATCH_SIZE, verbose=2)

    model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=2)

    y_pred = model.predict(x_test)

    helpers.print_precision_recall(y_pred, y_test)
    helpers.draw_confusion_matrix(y_pred, y_test)
    helpers.draw_plot(history)
    helpers.draw_roc_curve(y_pred, y_test, NUM_OF_LABELS)


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    data = pd.read_csv(FILE_NAME)

    if FEATURES_TYPE == 'band':
        x = selection.select_band_features(BAND_NAME, data)

    elif FEATURES_TYPE in ['channels_4', 'channels_6', 'channels_7', 'channels_8',
                           'channels_9', 'channels_10', 'channels_11', 'channels_28',
                           'channels_32', 'channels_42', 'channels_37', 'channels_50']:
        x = selection.select_channel_features(CHANNELS, data)

    else:
        x = data.iloc[:, 0:NUM_OF_FEATURES].to_numpy()

    y = data[LABEL_TYPE].to_numpy()

    if FEATURES_TYPE == 'anova':
        x = selection.select_anova_features(SELECTED_FEATURES, x, y)

    elif FEATURES_TYPE == 'chi2':
        x = selection.select_chi2_features(SELECTED_FEATURES, x, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=100, shuffle=True)

    model = create_model(x_train.shape[1])

    evaluate_model(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
