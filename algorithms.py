import numpy as np
import copy
import math
import cvxpy as cp


class PhasedElim():
    def __init__(self, arm, delta, max_l, dim=2, T=1000, iota=1):
        # todo: run off rounds instead of phases
        self.arm = np.array(arm)
        self.active = copy.deepcopy(self.arm)
        self.k = self.active.shape[0]
        self.delta = delta
        self.max_l = max_l
        self.l = 1
        self.dim = dim
        self.theta_hat = None
        self.T_l_a = None
        self.T_l = None
        self.T = []
        self.eliminated_arms = []
        self.mean_active_sets = [np.mean(self.active, axis=0)]
        self.theta_hats = []
        self.iota = iota

    def g_optimal_design(self):
        pi = cp.Variable(self.active.shape[0])
        def g(pi):
            m = self.active.shape[0]
            n = self.active.shape[1]
            V_pi = (sum([pi[i] * self.active[i, :].reshape(n, 1) @ cp.reshape(self.active[i,:], (1, n))
                       for i in range(m)]))
            return cp.log_det(V_pi)
        eps = 10**-5
        prob = cp.Problem(cp.Maximize(g(pi)), [pi <= np.ones_like(pi), cp.sum(pi) <= 1 + eps, cp.sum(pi) >= 1 - eps, pi >= np.zeros_like(pi)])
        prob.solve()
        return pi.value

    def compute(self):
        self.pi = self.g_optimal_design()
        # possibly a bug here...? it gave a negative once which may be an overflow if l is too large...
        # self.T_l_a = np.maximum(np.ceil(2 * self.dim * self.pi / 2**(-2 * self.l) * np.log(self.k * self.l * (self.l + 1) / self.delta)), 0)
        self.T_l_a = np.maximum(self.iota**2 * np.ceil(2**(2*self.l + 1) * self.dim * self.pi * np.log(self.k * self.l * (self.l + 1) / self.delta)), 0)
        self.T_l = np.sum(self.T_l_a)
        self.T.append(self.T_l)
        #print(np.max(self.T_l_a))
        return self.T_l_a

    def observe_rewards(self, rewards):
        V_l = sum([self.T_l_a[i] * self.active[i][:, np.newaxis] @ self.active[i][np.newaxis, :] for i in range(self.active.shape[0])])
        # V_l = np.sum(self.T_l_a[:, np.newaxis, np.newaxis] * (self.active[:, :, np.newaxis] @ self.active[:, np.newaxis, :]), axis=0)
        try:
            # maybe pinv?
            self.theta_hat = np.linalg.inv(V_l) @ np.sum(self.active * rewards[:, np.newaxis], axis=0)[:, np.newaxis]
            arms_to_keep = []
            eliminate = []
            for i, a in enumerate(self.active):
                diff = np.max((self.active - a[np.newaxis, :]) @ self.theta_hat)
                if diff <= 2 * 2**(-1 * self.l):
                    arms_to_keep.append(i)
                else:
                    eliminate.append(i)
            self.eliminated_arms.append(self.active[eliminate, :])
            self.active = self.active[arms_to_keep, :]
        except np.linalg.LinAlgError:
            print('Singular matrix detected. Repeating previous values')
            self.eliminated_arms.append(self.eliminated_arms[-1])
            return False
        self.k = self.active.shape[0]
        self.mean_active_sets.append(np.mean(self.active, axis=0))
        self.l += 1
        self.theta_hats.append(self.theta_hat)
        return True



class LinUCB():
    def __init__(self, arm, dim=2, lamda = 1, T=1000):
        self.arm = arm
        self.A_t = 0
        self.y = np.matrix(np.zeros((dim, 1)))
        self.V = np.matrix(lamda * np.identity(dim))
        self.lamda = lamda
        self.beta =  1 + np.sqrt(2*np.log(T)+dim*np.log((dim+T)/dim))
        self.actions = []
        self.T = T
        self.dim = dim
        self.action_idxs = []
        self.Vs = [copy.deepcopy(self.V)]
        self.debug_y = []
        self.debug_reward = []
        self.debug_y.append(self.y)
        self.debug_theta = []
        self.debug_theta.append(np.linalg.inv(self.V) * self.y)

    def observe_reward(self, reward):
        self.debug_reward.append(reward)
        self.V = self.V + self.arm[self.A_t].T * self.arm[self.A_t]
        self.Vs.append(copy.deepcopy(self.V))
        self.y = self.y + self.arm[self.A_t].T * reward
        self.debug_y.append(copy.deepcopy(self.y))
        self.debug_theta.append(copy.deepcopy(np.linalg.inv(self.V) * self.y))
        
    def compute(self):
        inv_V = np.linalg.inv(self.V)
        self.hat_theta = inv_V * self.y
        ucb = self.arm * self.hat_theta + self.beta * np.matrix(np.diag(np.sqrt( self.arm * inv_V * self.arm.T ))).T
        self.A_t = np.argmax(ucb)
        self.actions.append(self.arm[self.A_t])
        self.action_idxs.append(self.A_t)
        return self.arm[self.A_t]

    
class UCB:
    def __init__(self, action_set, dim=2, T=1000):
        self.arm = action_set
        self.T = T
        self.alpha = 1/(T**2)
        self.sample_means = np.asarray([0.0] * len(action_set))
        self.confidence_widths = np.asarray([np.inf] * len(action_set))
        self.num_pulls = np.asarray([0] * len(action_set))
        self.actions = []
        self.action_idxs = []
        self.dim = dim
        self.counter = 0
        self.debug_mean = {}
        self.debug_cw = {}
        self.debug_np = {}
        
    def compute(self):
        
        if self.counter < len(self.arm):
            index = self.counter
        else:
            index = np.argmax(self.sample_means + self.confidence_widths)
        
        self.debug_mean[self.counter] = copy.deepcopy(self.sample_means)
        self.debug_cw[self.counter] = copy.deepcopy(self.confidence_widths)
        self.debug_np[self.counter] = copy.deepcopy(self.num_pulls)
        self.counter += 1
        self.num_pulls[index] += 1
        
        self.action_idxs.append(index)
        self.actions.append(self.arm[index])
        return self.arm[index]
    
    def observe_reward(self, reward):
        
        cur_mean = self.sample_means[self.action_idxs[-1]]
        num_samples = self.num_pulls[self.action_idxs[-1]] - 1
        new_mean = (cur_mean * num_samples + reward)/(num_samples + 1)
        self.sample_means[self.action_idxs[-1]] = new_mean
        self.confidence_widths[self.action_idxs[-1]] = self.gcb(self.T, self.alpha, num_samples + 1)
        
    @staticmethod
    def gcb(T, alpha, num_samples):
        if num_samples == 0:
            return 2*T
        return np.sqrt(2*(math.pow(T, alpha) - 1)/(alpha * num_samples))
        
        
         
