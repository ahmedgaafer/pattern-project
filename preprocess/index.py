from preprocess.removeOutliers import remove_outliers
from preprocess.scale import scale
from preprocess.pca import pca

def pre_process(arr):
    arr = remove_outliers(arr)
    arr = pca(arr)
    arr = scale(arr)

    return arr

