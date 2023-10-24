from tensorflow import keras
from tensorflow.keras import layers


class Encoder(keras.layers.Layer):
    def __init__(self, hidden_units, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.enc_layers = [layers.LSTM(hidden_units, return_state=True, return_sequences=True) for _ in range(n_layers)]

    def call(self, x):
        for i in range(self.n_layers):
            x, state_h, state_c = self.enc_layers[i](x)
        return x, state_h, state_c


class Decoder(keras.layers.Layer):
    def __init__(self, hidden_units, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.dec_layers = [layers.LSTM(hidden_units, return_state=False, return_sequences=True) for _ in range(n_layers)]

    def call(self, x):
        encoder_outputs, decoder_input, state_h, state_c = x

        x = self.dec_layers[0](decoder_input, initial_state=[state_h, state_c])
        for i in range(1, self.n_layers):
            x = self.dec_layers[i](x)

        attention_outputs = layers.Attention()([x, encoder_outputs])
        x = keras.layers.Concatenate()([attention_outputs, x])

        return x


@keras.saving.register_keras_serializable()
class AttentionEncoderDecoder(keras.Model):
    def __init__(self, encoder_layers, decoder_layers, hidden_units, output_size):
        super().__init__()
        self.encoder = Encoder(hidden_units, encoder_layers)
        self.decoder = Decoder(hidden_units, decoder_layers)
        self.final_layer = layers.TimeDistributed(layers.Dense(output_size))

    def call(self, data):
        encoder_inputs, decoder_inputs = data
        encoder_outputs, state_h, state_c = self.encoder(encoder_inputs)
        x = self.decoder((encoder_outputs, decoder_inputs, state_h, state_c))
        x = self.final_layer(x)
        return x










