from tensorflow import keras

from transformer import Transformer
from encoder_decoder_wrapper import EncoderDecoderWrapper

MODEL_FILEPATH = "transformer_model.keras"
LOAD_MODEL = False
OUTPUT_TIMESTEPS = 4
BATCH_TIMESTEPS = 20
TEST_BATCHES = 10

if __name__ == '__main__':
    if LOAD_MODEL:
        transformer = keras.saving.load_model(MODEL_FILEPATH)
    else:
        # model arch
        n_layers = 4
        d_model = 12
        dff = 51
        num_heads = 2
        transformer = Transformer(n_layers, d_model, dff, num_heads, 1)

    # predictions
    wrapper = EncoderDecoderWrapper(transformer, OUTPUT_TIMESTEPS)
    wrapper.test(train_model=not LOAD_MODEL, batch_timesteps=BATCH_TIMESTEPS,
                 test_batches=TEST_BATCHES, model_filepath=MODEL_FILEPATH)



