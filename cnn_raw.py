import tensorflow as tf
import preprocessing_raw as prep
from sklearn.model_selection import train_test_split

TMIN, TMAX = 30, 90

signals, labels = prep.preprocess(TMIN, TMAX)

X_train, X_test, y_train, y_test = train_test_split(
    signals, labels, test_size=0.3, random_state=100)

n_features = X_train.shape[2]
n_channels = X_train.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(n_channels, n_features)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(13)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)
