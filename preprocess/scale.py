from sklearn.preprocessing import MinMaxScaler


def scale(arr):
    scaler = MinMaxScaler()
    scaler.fit(arr)
    print(" => Data was scaled to (0, 1)")
    return scaler.transform(arr)
