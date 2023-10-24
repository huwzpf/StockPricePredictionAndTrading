import tensorflow as tf
import numpy as np
from tensorflow import keras

from basic_data_prep import get_electricity_data
from basic_plots import plot_predictions


def split_input(data_batch, output_timesteps):
    return data_batch[:-output_timesteps]


def split_output(data_batch, output_timesteps):
    return data_batch[-output_timesteps:]


def split_input_target(data_batch, output_timesteps):
    # data_batch has BATCH_TIMESTEPS + output_timesteps timesteps
    # encoder input_size is BATCH_TIMESTEPS
    encoder_input = data_batch[:-output_timesteps]
    # decoder input size is output_timesteps, but it's first element is encoder input's last element
    decoder_input = data_batch[-(output_timesteps + 1): -1]
    # decoder output size is output_timesteps
    decoder_output = data_batch[-output_timesteps:]
    return (encoder_input, decoder_input), decoder_output


class EncoderDecoderWrapper(tf.Module):
    def __init__(self, model, output_timesteps):
        '''

        :param model: tf.keras.Model that when called takes [encoder_input, decoder_input] and return predictions
        :param output_timesteps: number ot steps to predict
        '''
        super().__init__()
        self.model = model
        self.output_timesteps = output_timesteps

    def __call__(self, x):
        '''
        :param x: input shaped batch_size x input_timesteps x input_channels
        :return: predictions shaped batch_size x output_timesteps x input_channels
        '''
        # x is
        output_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        start = np.zeros(x.shape[0])
        output_array = output_array.write(0, start)
        for i in range(self.output_timesteps):
            output = tf.transpose(output_array.stack()[tf.newaxis]) # batch_size x timesteps x output_size
            predictions = self.model([x, output], training=False) # batch_size x timesteps x output_size
            predicted_values = predictions[:, -1, -1]
            output_array = output_array.write(i + 1, predicted_values)

        return output_array.stack()[1:, :].numpy().transpose()

    def test(self, train_model, batch_timesteps, test_batches, model_filepath=None):
        train_dataset, validation_dataset, test_x, test_y = (
            get_electricity_data(split_x=lambda x: split_input(x, self.output_timesteps),
                                 split_y=lambda x: split_output(x, self.output_timesteps),
                                 split_input_target=lambda x: split_input_target(x, self.output_timesteps),
                                 batch_timesteps=batch_timesteps, test_batches=test_batches))
        if train_model:
            self.model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
            # fit
            self.model.fit(train_dataset, epochs=1, validation_data=validation_dataset)
            self.model.summary()
            # save model
            if model_filepath:
                self.model.save(model_filepath)

        test_outputs = self(test_x)
        test_real_vals = test_y[:, :, -1]
        test_inputs = test_x[:, :, -1]

        plot_predictions(batch_timesteps, self.output_timesteps,
                         test_batches, test_inputs, test_outputs, test_real_vals)
