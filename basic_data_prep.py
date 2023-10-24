import pandas as pd
import numpy as np
import tensorflow as tf
from normalize import normalize_array

n_locations = 40
n_samples = 25000


def get_electricity_data(split_x, split_y, split_input_target, batch_timesteps=25,
                         test_batches=20, validation_split_size=0.1, batch_size=10, buffer_size=10000):
    batches = []
    raw_data = pd.read_csv('electricity.txt', sep=',', header=None).to_numpy()[:n_samples, :n_locations].T

    for row in raw_data:
        row_reshaped = row.reshape(row.shape[0], 1)
        i = 0
        while i + batch_timesteps < len(row):
            batch_data = row_reshaped[i:i + batch_timesteps, :]
            batches.append(normalize_array(batch_data, 'min_max'))
            i += batch_timesteps

    np.random.shuffle(batches)

    test_x = np.array(list(map(split_x, batches[-test_batches:])))
    test_y = np.array(list(map(split_y, batches[-test_batches:])))

    dataset = tf.data.Dataset.from_tensor_slices(batches[:-test_batches])
    dataset = (
        dataset
        .map(split_input_target)
        .shuffle(buffer_size)
        .batch(batch_size))

    # saving VALIDATION_SIZE % of data for validation and one batch for testing after training
    validation_sample_size = int(len(dataset) * validation_split_size)
    validation_dataset = dataset.take(validation_sample_size)
    train_dataset = dataset.skip(validation_sample_size)

    return train_dataset, validation_dataset, test_x, test_y