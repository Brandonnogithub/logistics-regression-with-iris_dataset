import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data(train_size=0.8, seed=None, debug=False, debug_n=20):
    dataset = load_iris()
    data = dataset.data         # (150,4)
    target = dataset.target     # (150,1)
    n_samples = data.shape[0]   # smaple number (150)
    # split and shuffle data
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=train_size, test_size=1-train_size, random_state=seed)
    # weather use small size for debug
    if debug:
        debug_n = min(debug_n, n_samples)
        return x_train[:debug_n], x_test[:debug_n], y_train[:debug_n], y_test[:debug_n]

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    a,b,c,d = load_data(seed=20191225)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
    print(d)