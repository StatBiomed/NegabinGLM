# NB modified for NPC NLH data, with exposure (coef = 1)

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import gamma

def add_const(X: pd.DataFrame):
    X['const'] = 1
    return X

# negative binomial regression
class NB_Reg():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.params = np.ones((X.shape[1]), dtype=float)
        # phi = 1/alpha, dispersion parameter
        self.phi = 1

        self.optimize_record_w = []
        self.optimize_record_phi = []
    
    def log_likelihood_w_exp(self, params): #log likelihood of negative binomial distribution
        params = np.append(params, 1)
        mu = np.exp(np.dot(self.X, params))
        Y = self.Y
        phi = self.phi
        ll = np.sum(np.log((gamma(Y+phi)/((gamma(Y+1)*gamma(phi))))*((phi/(phi+mu))**phi)*((mu/(phi+mu))**Y)))
        # return negative log likelihood
        return np.sum(ll) * -1
    
    def log_likelihood_phi(self, phi): #log likelihood of negative binomial distribution
        mu = np.exp(np.dot(self.X, self.params))
        Y = self.Y
        ll = np.sum(np.log((gamma(Y+phi)/((gamma(Y+1)*gamma(phi))))*((phi/(phi+mu))**phi)*((mu/(phi+mu))**Y)))
        # return negative log likelihood
        return np.sum(ll) * -1

    def fit_w_exp(self):
        w = self.params[:-1]
        w = optimize.minimize(self.log_likelihood_w_exp, w, method='BFGS')

        self.optimize_record_w = w
        p = w.x
        p = np.append(p, 1)
        self.params = p
        return self

    def fit_phi(self):
        phi = self.phi
        phi = optimize.minimize(self.log_likelihood_phi, phi, method='BFGS')
        self.phi = phi.x
        self.optimize_record_phi = phi
        return self