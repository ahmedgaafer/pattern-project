import scipy.stats as stats
import numpy as np


def remove_outliers(arr):
    z_score = stats.zscore(arr)
    z_score_abs = np.abs(z_score)
    filtered_entries = (z_score_abs < 3).all(axis=1)
    new_arr = arr[filtered_entries]
    print(f" => {len(arr) - len(new_arr)} Outliers found and removed...")
    return new_arr
