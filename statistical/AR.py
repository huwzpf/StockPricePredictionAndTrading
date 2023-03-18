import pandas
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import pacf
from typing import List

'''
autoregressive model 
'''


def AR(q: int, data_vector: pandas.DataFrame, zero_coeffs: List[int] = None) -> np.ndarray:
    # get coefficients for linear regression using PACF (coeffs * X + b = Y)
    data_pacf = pacf(data_vector)
    coeffs = data_pacf[:q]
    if zero_coeffs:
        for zero_coeff in zero_coeffs:
            assert(zero_coeff < q)
            coeffs[zero_coeff] = 0

    # b = mean(coeffs * X - Y)
    res = data_vector.rolling(q).apply(lambda x: x.dot(coeffs))
    res_trimmed = res[q-1:-1].to_numpy()
    y = data_vector[q:].to_numpy()
    b = np.mean(y - res_trimmed)

    # calculate predictions
    preds = data_vector.rolling(q).apply(lambda x: x.dot(coeffs) + b)
    return preds[q:].to_numpy()


if __name__ == "__main__":
    # prepare data
    column = -2
    data_dir = 'data.csv'
    data = pd.read_csv(data_dir).iloc[:, column]
    data_vector = (data - np.mean(data))/np.std(data)

    q = 3
    y = data_vector[q:].to_numpy()
    preds = AR(q, data_vector, [0])

    # plot predictions
    plt.plot(preds, c="r")
    plt.plot(y, c='b')

    # calculate prediction error and correlation on diffs
    mse = np.mean(np.square(preds - y))
    print(f'MSE: {mse}')
    print(f'Correlaction on diffs: \n{np.corrcoef(np.diff(preds), np.diff(y))}')

    plt.show()
