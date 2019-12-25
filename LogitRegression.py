import numpy as np
from sag_solver import sag, log_dloss
from sklearn.metrics import accuracy_score


'''
z = w^T * x
y = 1 / (1+exp(-z))
loss = -Ylog(y) - (1-Y)log(1-y)     Y is the label
'''

def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def get_step_size(X, alpha, fit_intercept, classification=True):
    if classification:
        return (4.0 / (np.max(np.sum(X * X, axis=1)) +
                       fit_intercept + 4.0 * alpha))
    else:
        return 1.0 / (np.max(np.sum(X * X, axis=1)) + fit_intercept + alpha)


class LogitRegression_Li():
    def __init__(self, penalty='l2', tol=1e-4, C=1.0,
                 fit_intercept=True, random_state=None, solver='sag', max_iter=300):

        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        '''
        use training data to fit the weight
        '''
        y[y == 0] = -1
        alpha = 1.1
        step_size = get_step_size(X, alpha, self.fit_intercept)
        self.weights, self.intercept = sag(X, y, step_size, alpha, n_iter=self.max_iter,
                                   dloss=log_dloss,
                                   fit_intercept=self.fit_intercept,
                                   saga=self.solver == 'saga')
        y[y == -1] = 0

    def score(self, X, y):
        '''
        use weight to predit the label and given the acc socre
        '''
        z = np.dot(X, self.weights) + self.intercept
        z = Sigmoid(z)
        z[z > 0.5] = 1
        z[z < 0.6] = 0
        s = accuracy_score(z, y)
        return s

