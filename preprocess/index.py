from preprocess.removeOutliers import remove_outliers
from preprocess.scale import scale
from preprocess.pca import pca


def pre_process(arr, y=[], t="train"):
    if t == "train":
        arr, y = remove_outliers(arr, y, t=t)
    #arr = pca(arr)
    arr = scale(arr)

    return arr, y

