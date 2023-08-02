import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from normalize import normalize_array

output_timesteps = 5


def split_input_target(data_batch):
    input_data = data_batch[:-output_timesteps]
    target_data = data_batch[-output_timesteps:]
    return input_data, target_data


def split_only_input(data_batch):
    return data_batch[:-output_timesteps]


def split_only_target(data_batch):
    return data_batch[-output_timesteps:]


def dataset_to_numpy(d):
    return np.array(list(d.as_numpy_iterator())[0])

# This is simple autoencoder model
# It predicts output_steps = 5
# Decoder takes encoder output as input at each timestep
# It achieves less than 0.07 MSE on training set after 1 epoch

MODEL_FILEPATH = "autoenc_simple_model.keras"
LOAD_MODEL = False
# defines
n_locations = 40
n_samples = 25000
VALIDATION_SIZE = 0.02
BATCH_TIMESTEPS = 25
BATCH_SIZE = 5
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
        batch_data = row_reshaped[i:i+BATCH_TIMESTEPS+output_timesteps, :]
        batches.append(normalize_array(batch_data, 'min_max'))
        i += BATCH_TIMESTEPS + output_timesteps

np.random.shuffle(batches)

test_x = tf.data.Dataset.from_tensor_slices(batches[-BATCH_SIZE:]).map(split_only_input).batch(BATCH_SIZE)
test_y = tf.data.Dataset.from_tensor_slices(batches[-BATCH_SIZE:]).map(split_only_target).batch(BATCH_SIZE)

dataset = tf.data.Dataset.from_tensor_slices(batches[:-BATCH_SIZE])
dataset = (
    dataset
    .map(split_input_target)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE))

# saving VALIDATION_SIZE % of data for validation and one batch for testing after training
validation_sample_size = int(len(dataset) * VALIDATION_SIZE)
validation_dataset = dataset.take(validation_sample_size)
train_dataset = dataset.skip(validation_sample_size)

if LOAD_MODEL:
    model = tf.keras.saving.load_model(MODEL_FILEPATH)
else:
    #model arch
    hidden_units = 10

    encoder_inputs = layers.Input(shape=(BATCH_TIMESTEPS, 1))
    encoder = layers.LSTM(hidden_units, return_state=True, return_sequences=False, name="encoder")
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    decoder = layers.RepeatVector(output_timesteps)(tf.zeros((BATCH_SIZE, 1)))

    decoder_lstm = layers.LSTM(hidden_units, return_sequences=True, return_state=False, name="decoder")
    decoder_outputs = decoder_lstm(decoder, initial_state=[state_h, state_c])

    attention_outputs = layers.Attention()([decoder_outputs, encoder_outputs])
    dense_inputs = keras.layers.Concatenate()([attention_outputs, decoder_outputs])

    out = layers.Dense(1)(dense_inputs)
    model = models.Model(encoder_inputs, out)
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
    # fit
    model.fit(train_dataset, epochs=1, validation_data=validation_dataset)
    # save model
    model.save(MODEL_FILEPATH)

# model.summary()
test_outputs = model.predict(test_x)
test_real_vals = dataset_to_numpy(test_y)
test_inputs = dataset_to_numpy(test_x)

x_axis_data = np.arange(0, BATCH_TIMESTEPS, 1)
x_axis_pred = np.arange(BATCH_TIMESTEPS - 1, BATCH_TIMESTEPS + output_timesteps, 1)
for i in range(BATCH_SIZE):
    plt.figure()
    plt.plot(x_axis_data, test_inputs[i], color='blue', label='input')

    preds = test_inputs[i][-1].tolist() + test_outputs[i].reshape(-1).tolist()
    real = test_inputs[i][-1].tolist() + test_real_vals[i].reshape(-1).tolist()

    plt.plot(x_axis_pred, preds, color='red', label='prediction')
    plt.plot(x_axis_pred, real, color='green', label='real output')
    plt.legend()
    plt.show()
    print(f'Sample {i}:\nPredicted: {test_outputs[i]}, Real: {test_real_vals[i]}')

print(f'\nMSE ON TEST SET: {np.mean(np.square(test_outputs - test_real_vals))}')
