import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(batch_timesteps, output_timesteps, n_predictions, inputs, predictions, real_values):
    input_timesteps = batch_timesteps - output_timesteps
    x_axis_data = np.arange(0, input_timesteps, 1)
    x_axis_pred = np.arange(input_timesteps - 1, input_timesteps + output_timesteps, 1)
    for i in range(n_predictions):
        plt.figure()
        plt.plot(x_axis_data, inputs[i], color='blue', label='input')

        preds = [inputs[i][-1]] + predictions[i].tolist()
        real = [inputs[i][-1]] + real_values[i].tolist()

        plt.plot(x_axis_pred, preds, color='red', label='prediction')
        plt.plot(x_axis_pred, real, color='green', label='real output')
        plt.legend()
        plt.show()
        print(f'Sample {i}:\nPredicted: {predictions[i]}, Real: {real_values[i]}')

    print(f'\nMSE ON TEST SET: {np.mean(np.square(predictions - real_values))}')
