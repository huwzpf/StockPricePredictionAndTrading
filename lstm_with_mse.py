import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from normalize import normalize_array


def split_input_target(data_batch):
    input_data = data_batch[:-1]
    target_data = data_batch[1:]
    return input_data, target_data


def split_only_input(data_batch):
    return data_batch[:-1]


def split_only_target(data_batch):
    return data_batch[1:]


def dataset_to_numpy(d):
    return np.array(list(d.as_numpy_iterator())[0])

# This works a lot better than feed forward
# It achieves less than 0.03 MSE on training set with more than 0.75 correlation

MODEL_FILEPATH = "lstm_simple_model.keras"
LOAD_MODEL = False
# defines
n_locations = 40
n_samples = 25000
VALIDATION_SIZE = 0.02
BATCH_TIMESTEPS = 100
BATCH_SIZE = 3
# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000
#data prep
batches = []
raw_data = pd.read_csv('electricity.txt', sep=',', header=None).to_numpy()[:n_samples, :n_locations].T

for row in raw_data:
    row_reshaped = row.reshape(row.shape[0], 1)
    i = 0
    while i + BATCH_TIMESTEPS < len(row):
        batch_data = row_reshaped[i:i+BATCH_TIMESTEPS+1, :]
        batches.append(normalize_array(batch_data, 'min_max'))
        i += BATCH_TIMESTEPS + 1

test_x = tf.data.Dataset.from_tensor_slices(batches[-BATCH_SIZE:]).map(split_only_input).batch(BATCH_SIZE)
test_y = tf.data.Dataset.from_tensor_slices(batches[-BATCH_SIZE:]).map(split_only_target).batch(BATCH_SIZE)

dataset = tf.data.Dataset.from_tensor_slices(batches[:-BATCH_SIZE])
dataset = (
    dataset
    .map(split_input_target)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE))

validation_sample_size = int(len(dataset) * VALIDATION_SIZE)
validation_dataset = dataset.take(validation_sample_size)
train_dataset = dataset.skip(validation_sample_size)

if LOAD_MODEL:
    model = tf.keras.saving.load_model(MODEL_FILEPATH)
else:
    #model arch
    model = keras.Sequential([layers.LSTM(10, return_sequences=True), layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
    # fit
    model.fit(train_dataset, epochs=1, validation_data=validation_dataset)
    # save model
    model.save(MODEL_FILEPATH)

test_outputs = model.predict(test_x).reshape(-1)
test_real_vals = dataset_to_numpy(test_y).reshape(-1)

for i in range(20):
    print(f'Sample {i}:\nPredicted: {test_outputs[i]}, Real: {test_real_vals[i]}')

print(f'\nMSE ON TEST SET: {np.mean(np.square(test_outputs - test_real_vals))}')
