import numpy as np
import math
from scipy import linalg, sparse
from sklearn.datasets import make_blobs
from data_utils import load_data
from sklearn.metrics import accuracy_score

def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x)
    return exps / np.sum(exps)


def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically
    stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def softmax_dloss(z, y):
    '''
    dl/dzi = softmax(zi) - yi
    z : (3,)
    y : ()
    '''
    sz = softmax(z)
    l = len(sz)
    y_one_hot = np.zeros(l)
    y_one_hot[y] = 1
    return sz - y_one_hot
    

def log_dloss(p, y):
    z = p * y
    # approximately equal and saves the computation of the log
    if z > 18.0:
        return math.exp(-z) * -y
    if z < -18.0:
        return -y
    return -y / (math.exp(z) + 1.0)


def mat_time(x, y):
    a = x.shape[0]
    b = y.shape[0]
    c = np.zeros([a,b])
    for i in range(a):
        for j in range(b):
            c[i,j] = x[i] * y[j]
    return c


def sag(X, y, step_size, alpha, n_iter=1, dloss=None, sparse=False,
        fit_intercept=True, saga=False):
    n_samples, n_features = X.shape[0], X.shape[1]

    weights = np.zeros(X.shape[1])                      # (2,)
    sum_gradient = np.zeros(X.shape[1])                 # (2,)
    gradient_memory = np.zeros((n_samples, n_features)) # (20,2)

    intercept = 0.0
    intercept_sum_gradient = 0.0
    intercept_gradient_memory = np.zeros(n_samples)

    rng = np.random.RandomState(77)
    decay = 1.0
    seen = set()

    # sparse data has a fixed decay of .01
    if sparse:
        decay = .01

    for epoch in range(n_iter):
        for k in range(n_samples):
            idx = int(rng.rand(1) * n_samples)
            # idx = k
            entry = X[idx]
            seen.add(idx)
            p = np.dot(entry, weights) + intercept
            gradient = dloss(p, y[idx])
            update = entry * gradient + alpha * weights
            gradient_correction = update - gradient_memory[idx]
            sum_gradient += gradient_correction
            gradient_memory[idx] = update
            if saga:
                weights -= (gradient_correction *
                            step_size * (1 - 1. / len(seen)))

            if fit_intercept:
                gradient_correction = (gradient -
                                       intercept_gradient_memory[idx])
                intercept_gradient_memory[idx] = gradient
                intercept_sum_gradient += gradient_correction
                gradient_correction *= step_size * (1. - 1. / len(seen))
                if saga:
                    intercept -= (step_size * intercept_sum_gradient /
                                  len(seen) * decay) + gradient_correction
                else:
                    intercept -= (step_size * intercept_sum_gradient /
                                  len(seen) * decay)

            weights -= step_size * sum_gradient / len(seen)

    return weights, intercept


def sag_softmax(X, y, step_size, alpha, n_iter=1, dloss=None, sparse=False,
        fit_intercept=True, saga=False, n_class=3):
    n_samples, n_features = X.shape[0], X.shape[1]

    weights = np.zeros([n_features, n_class])                      # (4,3)
    sum_gradient = np.zeros([n_features, n_class])                 # (4,3)
    gradient_memory = np.zeros((n_samples, n_features, n_class))   # (20,4,3)

    intercept = np.zeros(n_class)
    intercept_sum_gradient = np.zeros(n_class)
    intercept_gradient_memory = np.zeros([n_samples, n_class])

    rng = np.random.RandomState(77)
    decay = 1.0
    seen = set()

    # sparse data has a fixed decay of .01
    if sparse:
        decay = .01

    for epoch in range(n_iter):
        for k in range(n_samples):
            idx = int(rng.rand(1) * n_samples)
            # idx = k
            entry = X[idx]  # (feature, )
            seen.add(idx)
            p = np.dot(entry, weights) + intercept  # (calss, )
            gradient = dloss(p, y[idx])             # (class, )
            update = mat_time(entry,  gradient) + alpha * weights   # (feature,class)
            gradient_correction = update - gradient_memory[idx]
            sum_gradient += gradient_correction
            gradient_memory[idx] = update
            if saga:
                weights -= (gradient_correction *
                            step_size * (1 - 1. / len(seen)))

            if fit_intercept:
                gradient_correction = (gradient -
                                       intercept_gradient_memory[idx])  # (class, )
                intercept_gradient_memory[idx] = gradient
                intercept_sum_gradient += gradient_correction
                gradient_correction *= step_size * (1. - 1. / len(seen))
                if saga:
                    intercept -= (step_size * intercept_sum_gradient /
                                  len(seen) * decay) + gradient_correction
                else:
                    intercept -= (step_size * intercept_sum_gradient /
                                  len(seen) * decay)

            weights -= step_size * sum_gradient / len(seen)

    return weights, intercept


def get_step_size(X, alpha, fit_intercept, classification=True):
    if classification:
        return (4.0 / (np.max(np.sum(X * X, axis=1)) +
                       fit_intercept + 4.0 * alpha))
    else:
        return 1.0 / (np.max(np.sum(X * X, axis=1)) + fit_intercept + alpha)



def test_classifier_matching():
    n_samples = 20
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=0,
                      cluster_std=0.1)
    # print(X)
    # print(y)
    y[y == 0] = -1
    alpha = 1.1
    fit_intercept = True
    step_size = get_step_size(X, alpha, fit_intercept)  # 0.1528
    for solver in ['sag', 'saga']:
        if solver == 'sag':
            n_iter = 80
        else:
            # SAGA variance w.r.t. stream order is higher
            n_iter = 300
        weights, intercept = sag(X, y, step_size, alpha, n_iter=n_iter,
                                   dloss=log_dloss,
                                   fit_intercept=fit_intercept,
                                   saga=solver == 'saga')
        weights = np.atleast_2d(weights)
        intercept = np.atleast_1d(intercept)
        print(weights)
        print(intercept)


def test_sag_sofmax():
    X, xx, y, yy = load_data(seed=20191225, two_class=False)
    alpha = 1.1
    fit_intercept = True
    step_size = get_step_size(X, alpha, fit_intercept)
    n_iter = 100
    weights, intercept = sag_softmax(X, y, step_size, alpha, n_iter=n_iter,
                                   dloss=softmax_dloss,
                                   fit_intercept=fit_intercept,
                                   saga=False, n_class=3)
    z = np.dot(xx, weights) + intercept
    n = z.shape[0]
    for i in range(n):
        z[i] = softmax(z[i])
    argmax = lambda x:np.argmax(x, 1)
    pred = argmax(z)
    print(pred)
    print(yy)
    print(accuracy_score(pred, yy))


if __name__ == "__main__":
    # test_classifier_matching()
    test_sag_sofmax()
