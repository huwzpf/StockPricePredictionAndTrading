import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from normalize import normalize_array
MODEL_FILEPATH = "transformer_model.keras"
LOAD_MODEL = False
# defines
n_locations = 40
n_samples = 25000
VALIDATION_SIZE = 0.02
BATCH_TIMESTEPS = 25
BATCH_SIZE = 5
TEST_BATCHES = BATCH_SIZE
# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000
output_timesteps = 5


def split_encoder(data_batch):
    return data_batch[:-output_timesteps]


def split_output(data_batch):
    return data_batch[-output_timesteps:]


def split_input_target(data_batch):
    # data_batch has BATCH_TIMESTEPS + output_timesteps timesteps
    # encoder input_size is BATCH_TIMESTEPS
    encoder_input = data_batch[:-output_timesteps]
    # decoder input size is output_timesteps, but it's first element is encoder input's last element
    decoder_input = data_batch[-(output_timesteps + 1): -1]
    # decoder output size is output_timesteps
    decoder_output = data_batch[-output_timesteps:]
    return (encoder_input, decoder_input), decoder_output


def postional_encoding(timesteps, d_model):
    # need to define max_timesteps since tensorflow complains about timesteps variable being tensor
    max_timesteps = 2048
    # output of embedding layer is sized d_model x timesteps
    # let's create matrix that will be added to it
    positions = np.arange(max_timesteps)[:, np.newaxis]  # [0, 1, ...] casted to shape timesteps x 1
    depths = np.arange(0, d_model, 2)[np.newaxis,
             :] / d_model  # [0/d_model, 2/d_model, ...] casted to shape 1 x d_model/2
    angle_rates = positions / (10000 ** depths)  # timesteps x d_model/2

    encoding = np.zeros((max_timesteps, d_model))
    encoding[:, 0::2] = np.sin(angle_rates)
    # if d_model is odd, cosine function will be used 1 time less than sine
    if d_model % 2 == 0:
        encoding[:, 1::2] = np.cos(angle_rates)
    else:
        encoding[:, 1::2] = np.cos(angle_rates)[:, :-1]

    return tf.cast(encoding, dtype=tf.float32)[:timesteps, :]

class TransformerInput(keras.layers.Layer):
  def __init__(self, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = keras.layers.Dense(d_model)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    x = x + postional_encoding(length, self.d_model)
    return x

class FeedForward(keras.layers.Layer):
    def __init__(self, d_model, dff):
        super().__init__()
        self.seq = keras.Sequential([
          keras.layers.Dense(dff, activation='relu'),
          keras.layers.Dense(d_model),
        ])
    def call(self, x):
        x = self.seq(x)
        return x


class AddNormLayer(keras.layers.Layer):
    def __init__(self, sublayer):
        # this class implements residual connection around sublayer
        # along with normalization layer
        super().__init__()
        self.sublayer = sublayer
        self.layernorm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()

    def call(self, x, *args):
        sublayer_output = self.sublayer(x, *args)
        x = self.add([x, sublayer_output])  # residual connection
        x = self.layernorm(x)
        return x


class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.self_attention = AddNormLayer(lambda x: self.mha(query=x, value=x, key=x))
        self.ffn = AddNormLayer(FeedForward(d_model, dff))

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(keras.layers.Layer):
    def __init__(self, n_layers, d_model, num_heads, dff):
        super().__init__()
        self.n_layers = n_layers
        self.input_layer = TransformerInput(d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff) for _ in range(n_layers)]

    def call(self, x):
        x = self.input_layer(x)
        for i in range(self.n_layers):
            x = self.enc_layers[i](x)

        return x


class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.self_attention = AddNormLayer(lambda x: self.mha(query=x, value=x, key=x, use_causal_mask=True))
        self.cross_attention = AddNormLayer(lambda x, context: self.mha(query=x, value=context, key=context))
        self.ffn = AddNormLayer(FeedForward(d_model, dff))

    def call(self, x, context):
        x = self.self_attention(x)
        x = self.cross_attention(x, context)
        x = self.ffn(x)
        return x


class Decoder(keras.layers.Layer):
    def __init__(self, n_layers, d_model, num_heads, dff):
        super().__init__()
        self.n_layers = n_layers
        self.input_layer = TransformerInput(d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff) for _ in range(n_layers)]

    def call(self, x, context):
        x = self.input_layer(x)
        for i in range(self.n_layers):
            x = self.dec_layers[i](x, context)

        return x

@keras.saving.register_keras_serializable()
class Transformer(keras.Model):
    def __init__(self, n_layers, d_model, num_heads, dff, output_size):
        super().__init__()
        self.encoder = Encoder(n_layers, d_model, num_heads, dff)
        self.decoder = Decoder(n_layers, d_model, num_heads, dff)
        self.final_layer = keras.layers.Dense(output_size)

    def call(self, data):
        encoder_inputs, decoder_inputs = data # decoder inputs are outputs shifted right
        context = self.encoder(encoder_inputs)
        x = self.decoder(decoder_inputs, context)
        x = self.final_layer(x)
        return x


class TransformerWrapper(tf.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def __call__(self, x):
        output_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        start = np.zeros(x.shape[0])
        output_array = output_array.write(0, start)
        for i in range(output_timesteps):
            output = tf.transpose(output_array.stack()[tf.newaxis]) # batch_size x timesteps x output_size
            predictions = transformer([x, output], training=False) # batch_size x timesteps x output_size
            predicted_values = predictions[:, -1, -1]
            output_array = output_array.write(i + 1, predicted_values)

        return output_array.stack()[1:, :].numpy()

# This is basic Transformer model
# It predicts output_steps = 5
# Decoder takes encoder output as input at each timestep
# It achieves less than 0.015 MSE on training set after 1 epoch


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

test_x = np.array(list(map(split_encoder, batches[-TEST_BATCHES:])))
test_y = np.array(list(map(split_output, batches[-TEST_BATCHES:])))

dataset = tf.data.Dataset.from_tensor_slices(batches[:-TEST_BATCHES])
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
    transformer = keras.saving.load_model(MODEL_FILEPATH)
else:
    #model arch
    n_layers = 4
    d_model = 12
    dff = 51
    num_heads = 2

    transformer = Transformer(n_layers, d_model, dff, num_heads, 1)
    transformer.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
    # fit
    transformer.fit(train_dataset, epochs=1, validation_data=validation_dataset)
    transformer.summary()
    # save model
    transformer.save(MODEL_FILEPATH)

# predictions

t = TransformerWrapper(transformer)
test_outputs = t(test_x)
test_real_vals = test_y[:, :, -1]
test_inputs = test_x[:, :, -1]

x_axis_data = np.arange(0, BATCH_TIMESTEPS, 1)
x_axis_pred = np.arange(BATCH_TIMESTEPS - 1, BATCH_TIMESTEPS + output_timesteps, 1)
for i in range(TEST_BATCHES):
    plt.figure()
    plt.plot(x_axis_data, test_inputs[i], color='blue', label='input')

    preds = [test_inputs[i][-1]] + test_outputs[i].tolist()
    real = [test_inputs[i][-1]] + test_real_vals[i].tolist()

    plt.plot(x_axis_pred, preds, color='red', label='prediction')
    plt.plot(x_axis_pred, real, color='green', label='real output')
    plt.legend()
    plt.show()
    print(f'Sample {i}:\nPredicted: {test_outputs[i]}, Real: {test_real_vals[i]}')

print(f'\nMSE ON TEST SET: {np.mean(np.square(test_outputs - test_real_vals))}')
