import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from normalize import normalize_array

# Defines
LOAD_MODEL = False
MODEL_FILEPATH = "lstm_model_quotes.keras"
DATE = 0
PRICE = 1

# Parameters
sequence_length = 50
samples_n = 200
batch_size = 3
training_size = 0.8
test_size = 1 - training_size
epochs = 50

# (0) Plot actual quotes data
df = pd.read_csv('datasets/quotes/BP_quotes.csv', sep=',', header=None)
df = df.drop(df.index[0])
df = df.iloc[::-1]
df = df.reset_index(drop=True)
df[PRICE] = pd.to_numeric(df[PRICE].str.replace('$', '')).astype(float)
df[DATE] = pd.to_datetime(df[DATE])
plt.figure(figsize=(10, 6)) 
plt.xlabel('Time')
plt.ylabel('Price($)')
plt.title('BP')
plt.grid(True)
plt.plot(df[DATE],df[PRICE])
plt.tight_layout()
plt.show()

# (1) Data preparation
data = df[PRICE][:samples_n].to_numpy()
sequences = []
targets = []
for i in range(len(data) - sequence_length):
    sequences.append(data[i:i+sequence_length])
    targets.append(data[i+sequence_length])

sequences = np.array(sequences)
targets = np.array(targets)

normalize_array(sequences, 'min_max')
normalize_array(targets, 'min_max')

# (2) Split
split_index = int(training_size * len(sequences))

X_train, X_test = sequences[:split_index], sequences[split_index:]
y_train, y_test = targets[:split_index], targets[split_index:]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)

# (3) Architecture
if LOAD_MODEL:
    model = tf.keras.saving.load_model(MODEL_FILEPATH)
else:
    model = keras.Sequential()
    model.add(layers.LSTM(64, input_shape=(sequence_length, 1)))
    model.add(layers.Dense(1)) 
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(sequences, targets, epochs=epochs, batch_size=batch_size) 
    model.save(MODEL_FILEPATH)

# (4) Test
    model.summary()
    test_outputs = model.predict(X_test).reshape(-1)
    test_real_vals = y_test.reshape(-1)

for i in range(20):
    print(f'Sample {i}:\nPredicted: {test_outputs[i]}, Real: {test_real_vals[i]}')

print(f'\nMSE ON TEST SET: {np.mean(np.square(test_outputs - test_real_vals))}')