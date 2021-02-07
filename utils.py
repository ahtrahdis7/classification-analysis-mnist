import numpy as np
from sklearn.model_selection import train_test_split


def process_data(data):
    print("pre-processing data")
    X = np.array(data[0][0])
    y = np.array(data[0][1])
    X_flat = []
    for i in range(len(X)):
        X_flat.append(X[i].flatten())
    X = np.array(X_flat)
    # X_flat.shape
    return train_test_split(X,y)