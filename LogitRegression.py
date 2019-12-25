import numpy as np
from sag_solver import sag, log_dloss, sag_softmax, softmax_dloss, softmax
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
                 fit_intercept=True, random_state=None, solver='sag', max_iter=300, n_class=2):

        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.argmax = lambda x:np.argmax(x, 1)
        self.n_class = n_class

    def fit_bi(self, X, y):
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

    def fit_multi(self, X, y):
        n_class = np.max(y) + 1     # class number
        alpha = 1.1
        step_size = get_step_size(X, alpha, self.fit_intercept)
        n_iter = 100
        self.weights, self.intercept = sag_softmax(X, y, step_size, alpha, n_iter=n_iter,
                                   dloss=softmax_dloss,
                                   fit_intercept=self.fit_intercept,
                                   saga=False, n_class=n_class)

    def fit(self, X, y):
        if self.n_class == 2:
            return self.fit_bi(X, y)
        else:
            return self.fit_multi(X, y)

    def predict_bi(self, X):
        z = np.dot(X, self.weights) + self.intercept
        z = Sigmoid(z)
        z[z > 0.5] = 1
        z[z < 0.6] = 0
        return z

    def predict_softmax(self, X):
        z = np.dot(X, self.weights) + self.intercept
        n = z.shape[0]
        for i in range(n):
            z[i] = softmax(z[i])
        pred = self.argmax(z)
        return pred

    def predict(self, X):
        if self.n_class == 2:
            return self.spredict_bi(X)
        else:
            return self.predict_softmax(X)

    def score(self, X, y):
        pred = self.predict(X)
        s = accuracy_score(y, pred)
        return s
