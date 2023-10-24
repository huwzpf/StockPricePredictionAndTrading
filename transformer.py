import numpy as np
import tensorflow as tf
from tensorflow import keras


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
        encoder_inputs, decoder_inputs = data
        context = self.encoder(encoder_inputs)
        x = self.decoder(decoder_inputs, context)
        x = self.final_layer(x)
        return x
