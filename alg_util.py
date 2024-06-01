import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gennorm
from tqdm import tqdm

def train_alg_UCB(alg, T, theta, oracle):
    for time in range(T):
        action = alg.compute()
        reward = oracle.compute_reward(action)
        alg.observe_reward(reward)

def train_alg_Phased_Elim(alg, theta, oracle):
    max_l = 0
    for phase in tqdm(range(alg.max_l)):
        action_counts = alg.compute().astype(np.int64)
        rewards = np.zeros_like(action_counts).astype(np.float64)
        for i, action in enumerate(alg.active):
            # for _ in range(action_counts[i]):
            rewards[i] = np.sum(oracle.compute_reward(action, size=action_counts[i]))
        rtn = alg.observe_rewards(rewards)
        max_l += 1
        if not rtn:
            break
    return max_l


def generate_k_dim_vector(dim=2):
    full_index = random.randint(0,dim-1)
    vec = np.random.random(int(dim)) * 2 - 1
    vec[full_index] = np.sign(vec[full_index]) * 1
    return vec

def generate_random_vec(dim=2, mag=1, ord=2):
    vec = gennorm.rvs(ord, size=dim)
    vec = vec/np.linalg.norm(vec, ord=ord)*mag
    return vec

def generate_random_l1_vec(dim=2, mag=1) -> np.ndarray:
    vec = np.random.random(int(dim))
    vec = vec/np.linalg.norm(vec, ord=1)*mag
    return vec

def generate_random_linf_vec(dim=2, mag=1) -> np.ndarray:
    vec = np.random.random(int(dim))
    vec = vec/np.linalg.norm(vec, ord=np.inf)*mag
    return vec

def plot_vectors(theta, actions):
    plt.axis([-1, 1, -1, 1])
    plt.quiver(0, 0, theta[0], theta[1], color='green', scale=1)
    for idx, action in enumerate(actions):
        plt.quiver(0, 0, action[0], action[1], color = [(idx/len(actions), 0, 0)], scale=1)
