import numpy as np

def normalize_array(arr, method='min_max'):
    if method == 'min_max':
        # Min-max normalization
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        denominator = 1 if arr_min == arr_max else (arr_max - arr_min)
        normalized_arr = (arr - arr_min) / denominator
    elif method == 'mean_std':
        # Mean and standard deviation normalization
        arr_mean = np.mean(arr)
        arr_std = np.std(arr)
        normalized_arr = (arr - arr_mean) / arr_std
    else:
        raise ValueError("Invalid normalization method. Choose either 'min_max' or 'mean_std'.")

    return normalized_arr