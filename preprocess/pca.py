from sklearn.decomposition import PCA
import numpy as np

def pca(arr):
    reducer = 3
    arr = np.array(arr)
    p = PCA(n_components=reducer)
    p.fit(arr)
    print(f" => [PCA applied] Reduced the number of components to ( {reducer} ) with score: { p.singular_values_}", end="\n")
    return p.transform(arr)