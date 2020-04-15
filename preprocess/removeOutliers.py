import scipy.stats as stats
import numpy as np


def remove_outliers(arr, y, t="train"):
    z_score = stats.zscore(arr)
    z_score_abs = np.abs(z_score)
    filtered_entries = (z_score_abs < 3).all(axis=1)
    new_arr = arr[filtered_entries]
    print(f" => {len(arr) - len(new_arr)} Outliers found and removed...")
    if t == "train":
        new_y = y[filtered_entries]
        return new_arr, new_y
    return new_arr
