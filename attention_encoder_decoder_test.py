from tensorflow import keras

from attention_encoder_decoder import AttentionEncoderDecoder
from encoder_decoder_wrapper import EncoderDecoderWrapper

MODEL_FILEPATH = "attention_encoder_decoder_model.keras"
LOAD_MODEL = False
OUTPUT_TIMESTEPS = 7
BATCH_TIMESTEPS = 40
TEST_BATCHES = 8

if __name__ == '__main__':
    if LOAD_MODEL:
        encoder_decoder = keras.saving.load_model(MODEL_FILEPATH)
    else:
        # model arch
        encoder_layers = 2
        decoder_layers = 2
        hidden_units = 50

        encoder_decoder = AttentionEncoderDecoder(encoder_layers, decoder_layers, hidden_units, 1)

        wrapper = EncoderDecoderWrapper(encoder_decoder, OUTPUT_TIMESTEPS)
        wrapper.test(train_model=not LOAD_MODEL, batch_timesteps=BATCH_TIMESTEPS,
                     test_batches=TEST_BATCHES, model_filepath=MODEL_FILEPATH)



