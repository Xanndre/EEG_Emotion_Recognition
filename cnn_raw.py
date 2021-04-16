import tensorflow as tf
import helpers
import preprocessing_raw as prep
from sklearn.model_selection import train_test_split

TMIN, TMAX = 30, 90
NUM_OF_EPOCHS = 10

signals, labels = prep.preprocess(TMIN, TMAX)

x_train, x_test, y_train, y_test = train_test_split(
    signals, labels, test_size=0.3, random_state=100)

n_features = x_train.shape[2]
n_channels = x_train.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(n_channels, n_features)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(13)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=NUM_OF_EPOCHS,
                    validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

helpers.draw_plot(history)

print('\nTest accuracy:', test_acc)
