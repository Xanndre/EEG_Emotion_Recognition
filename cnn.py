import pandas as pd
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import train_test_split


def evaluate_model(x_train, y_train, x_test, y_test):
    verbose, epochs, batch_size = 0, 200, 32
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs,
              batch_size=batch_size, verbose=verbose)

    _, accuracy = model.evaluate(
        x_test, y_test, batch_size=batch_size, verbose=0)
    return accuracy


def main():
    data = pd.read_csv("features.csv")

    x = data.iloc[:, 0:216].to_numpy()
    y = data.iloc[:, -1:].to_numpy()

    x = x.reshape((x.shape[0], x.shape[1], 1))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=100)

    accuracy = evaluate_model(x_train, y_train, x_test, y_test)
    print('Accuracy: ', accuracy)


if __name__ == "__main__":
    main()
