import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import pacf

'''
autoregressive model 
'''

if __name__ == "__main__":
    # prepare data
    column = -2
    data_dir = 'data.csv'
    data = pd.read_csv(data_dir).iloc[:, column]
    data_vector = (data - np.mean(data))/np.std(data)

    # get coefficients for linear regression using PACF (coeffs * X + b = Y)
    data_pacf = pacf(data_vector)
    coeffs = data_pacf[:5]

    # print(coeffs)
    # [ 1.          0.99510463 -0.0209996   0.03668808 -0.02132917]
    # first coefficient is discarded

    coeffs[:1] = [0]

    # b = mean(coeffs * X - Y)
    res = data_vector.rolling(5).apply(lambda x: x.dot(coeffs))
    res_trimmed = res[4:-1].to_numpy()
    y = data_vector[5:].to_numpy()
    b = np.mean(y - res_trimmed)

    # calculate predictions
    preds = data_vector.rolling(5).apply(lambda x: x.dot(coeffs) + b)[5:].to_numpy()

    # plot predictions
    plt.plot(preds, c="r")
    plt.plot(y, c='b')

    # calculate prediction error and correlation on diffs
    mse = np.mean(np.square(preds - y))
    print(f'MSE: {mse}')
    print(f'Correlaction on diffs: \n{np.corrcoef(np.diff(preds), np.diff(y))}')

    plt.show()
