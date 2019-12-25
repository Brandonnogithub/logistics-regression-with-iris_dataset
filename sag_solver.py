import numpy as np
import math
from scipy import linalg, sparse
from sklearn.datasets import make_blobs


def log_dloss(p, y):
    z = p * y
    # approximately equal and saves the computation of the log
    if z > 18.0:
        return math.exp(-z) * -y
    if z < -18.0:
        return -y
    return -y / (math.exp(z) + 1.0)


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


if __name__ == "__main__":
    test_classifier_matching()
