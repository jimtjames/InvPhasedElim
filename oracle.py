import numpy as np
from scipy.stats import truncnorm

class Oracle:
    def __init__(self, theta, upper=1, lower=-1, sigma=0.2):
        self.theta = theta
        self.upper = upper
        self.lower = lower
        self.sigma = sigma

    def compute_reward(self, action, size=1):
        mu = np.inner(action, self.theta)
        if self.sigma <= 1e-5:
            return np.ones(size) * mu
        X = truncnorm((self.lower) / self.sigma, (self.upper) / self.sigma, loc=mu, scale=self.sigma)
        # X = truncnorm((self.lower - mu) / self.sigma, (self.upper - mu) / self.sigma, loc=mu, scale=self.sigma)
        rewards = X.rvs(size)
        if size == 1:
            return rewards[0]
        return rewards

def calc_pseudoregret(theta, actions):
    new_actions = np.stack(actions, axis=0)
    return -np.sum(np.dot(new_actions, theta))
