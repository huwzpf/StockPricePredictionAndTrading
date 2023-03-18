import pandas
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import List
from utlis import download_csv

'''
moving average model 
'''
# TODO check correctness


def MA(q: int, data_vector: pandas.DataFrame, zero_coeffs:  List[int] = None) -> np.ndarray:
    rolling_avg = data_vector.rolling(q).mean()
    errors = data_vector - rolling_avg
    x = np.column_stack([errors.shift(i) for i in range(q) if i not in zero_coeffs])

    preds_start = (q - 1) * 2
    y = data_vector[preds_start:]
    x = x[preds_start:]

    # calculate coeffs using least squares
    coeffs = np.linalg.lstsq(x, y)[0]
    preds = x.dot(coeffs) + rolling_avg[preds_start:]

    return preds.to_numpy()


if __name__ == "__main__":
    # prepare data
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00409/Daily_Demand_Forecasting_Orders.csv'
    df = download_csv(data_url, delimiter=';')
    data = pd.to_numeric(df['Banking orders (2)']).dropna()
    # data_vector = (data - np.mean(data))/np.std(data)
    data_vector = data

    q = 10
    preds_start = (q - 1) * 2
    y = data_vector[preds_start:].to_numpy()
    preds = MA(q, data_vector, [0, 1, 2, 4, 5, 6, 7, 8])

    # plot predictions
    plt.plot(preds, c="r")
    plt.plot(y, c='b')

    # calculate prediction error and correlation on diffs
    mse = np.mean(np.square(preds - y))
    print(f'MSE: {mse}')
    print(f'Correlaction on diffs: \n{np.corrcoef(np.diff(preds), np.diff(y))}')

    plt.show()