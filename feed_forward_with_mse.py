import numpy as np
import pandas as pd
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# this does not work very well xD
# after 30 iters goes to MSE ~= 0.8 and correlation ~= 0.44 while training,
# although it does not generalize too well - validation loss is kept usually around 0.95 with correlation of ~0.1
# which results in mean absolute error on test set of around 0.73

# not sure if it's overfitting or there's something wrong with data prep / network arch
# or just acceptable generalization results can't be achieved with feed-forward NN

# defines
n_locations = 40
n_samples = 25000
n_val = 200
n_test = 50
window_size = 24 * 7
#data prep
raw_data = pd.read_csv('electricity.txt', sep=',', header=None).to_numpy()[:n_samples, :n_locations]
mean = np.mean(raw_data, axis=0, keepdims=True)
std = np.std(raw_data, axis=0, keepdims=True)
data = (raw_data - mean)/std
np.random.shuffle(data)
X_raw = X = np.lib.stride_tricks.sliding_window_view(data[:-1], (window_size, 1))
X = X_raw.reshape(-1, window_size)
Y_raw =  data[window_size:]
Y = Y_raw.reshape(-1)
# split data
x_train = X[n_val + n_test:]
y_train = Y[n_val + n_test:]
x_val = X[n_test:n_val + n_test]
y_val = Y[n_test:n_val + n_test]
x_test = X[:n_test]
y_test = Y[:n_test]
# network arch
model = keras.Sequential([keras.Input(shape=(window_size,)),
                          layers.Dense(40, activation="relu"),
                          layers.Dense(80, activation="relu"),
                          layers.Dense(160, activation="relu"),
                          layers.Dense(320, activation="relu"),
                          layers.Dense(80, activation="relu"),
                          layers.Dense(20, activation="relu"),
                          layers.Dense(1, activation="linear")])
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError(), metrics=[tfp.stats.correlation])

# fit
model.fit(x_train, y_train, batch_size=100, epochs=30, validation_data=(x_val, y_val))

# test
preds = model.predict(x_test)
test_diffs = []
for i in range(n_test):
    test_diff = tf.math.abs(preds[i] - y_test[i])
    test_diffs.append(test_diff)
    print(f'TEST{i}:\nPrediction- {preds[i]}, Reality- {y_test[i]}, ABS(Diff)- {test_diff}')

print(f'MEAN - {np.mean(test_diffs)}')
