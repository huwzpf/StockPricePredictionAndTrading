import pandas
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import pacf
from typing import List
from utlis import download_csv

'''
autoregressive model 
'''


def AR(q: int, data_vector: pandas.DataFrame, zero_coeffs: List[int] = None) -> np.ndarray:
    # get coefficients for linear regression using PACF (coeffs * X + b = Y)
    data_pacf = pacf(data_vector)[:q]
    coeffs  = [coeff if idx not in zero_coeffs else 0 for idx, coeff in enumerate(data_pacf)]
    coeffs = list(reversed(coeffs))

    # b = mean(coeffs * X - Y)
    res = data_vector.rolling(q).apply(lambda x: x.dot(coeffs))
    res_trimmed = res[q-1:-1].to_numpy()
    y = data_vector[q:].to_numpy()
    b = np.mean(y - res_trimmed)
    print(f'Fitted coefficients : A: {coeffs}, B: {b}')
    # calculate predictions
    preds = data_vector.rolling(q).apply(lambda x: x.dot(coeffs) + b)
    return preds[q:].to_numpy()


if __name__ == "__main__":
    # prepare data
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00409/Daily_Demand_Forecasting_Orders.csv'
    df = download_csv(data_url, delimiter=';')
    data = pd.to_numeric(df['Banking orders (2)']).dropna()
    # data_vector = (data - np.mean(data))/np.std(data)
    data_vector = data

    q = 4
    y = data_vector[q:].to_numpy()
    preds = AR(q, data_vector, [0, 1])

    # plot predictions
    plt.plot(preds, c="r")
    plt.plot(y, c='b')

    # calculate prediction error and correlation on diffs
    mse = np.mean(np.square(preds - y))
    print(f'MSE: {mse}')
    print(f'Correlaction on diffs: \n{np.corrcoef(np.diff(y), np.diff(preds))}')
    plt.show()
