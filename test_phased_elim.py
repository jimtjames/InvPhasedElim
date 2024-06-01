import os
import pickle
from threading import local
import numpy as np
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.app.app import python_app
from sampler import SyntheticSampler
from tqdm import tqdm
import parsl

# multi_site_config = Config(
    # executors=[
        # HighThroughputExecutor(
            # label="local_htex",
            # worker_debug=True,
            # max_workers=32,
            # provider=LocalProvider(
                # channel=LocalChannel(),
                # init_blocks=1,
                # max_blocks=1,
            # ),
        # )
    # ],
# )
local_config = Config(
    executors=[
        ThreadPoolExecutor(max_threads=1)
    ]
)

parsl.load(local_config)

working_directory = os.getcwd()


def run_amzn_test():
    import sys
    import os
    sys.path.append(os.path.abspath(working_directory))
    from sampler import SyntheticSampler
    from test_util import test_Phased_Elim
    from alg_util import generate_random_vec
    from estimators import estimate_phased_elim_means, estimate_phased_elim_means_slow
    import numpy as np
    vals = {'fwd': [], 'invf': [], 'inv': [], 'rand': []}
    ord = 2
    for j in range(2, 9, 2):
        # sampler = SyntheticSampler(2000, dim=10, ord=ord)
        # theta, action_set, sigma = sampler.sample()
        # KEEP BELOW
        user_factors = np.load('data/amzn/user_factors_%d.npy' % j)
        item_factors = np.load('data/amzn/item_factors_%d.npy' % j)
        user_factors = user_factors[user_factors.any(axis=1)]
        item_factors = item_factors[item_factors.any(axis=1)]
        choice = np.random.randint(0, user_factors.shape[0])
        movie_choice = np.random.randint(0, item_factors.shape[0], size=6000)
        action_set = item_factors[movie_choice]
        theta = user_factors[choice:choice+1]
        # print(theta.shape)
        # KEEP ABOVE
        sigma = 0.2
        # sampler = SyntheticSampler(2000, dim=d, ord=ord)
        # theta, action_set, sigma = sampler.sample()
        R = np.max(action_set @ theta.T)
        a_star = action_set[np.argmax(action_set @ theta.T)][np.newaxis, :]#[:, np.newaxis]
        # print(a_star.shape)
        # print(R)
        max_l = 5
        fwd_theta, _, alg, max_l = test_Phased_Elim(theta, action_set, sigma, max_l=max_l, iota=0.8)
        # for i in range(, max_l):
        fwd_theta = alg.theta_hats[-1].T
        fwd_theta_norm = fwd_theta / np.linalg.norm(fwd_theta, ord=1)
        theta_norm = theta / np.linalg.norm(theta, ord=1)
        vals['fwd'].append(np.linalg.norm(fwd_theta_norm - theta_norm) / np.linalg.norm(theta_norm))
        print(vals['fwd'][-1])
        for i in range (max_l-1, 1, -1):
            inv_error, inv_theta = estimate_phased_elim_means_slow(alg, a_star, R, theta, l=max_l - 3)
            if inv_theta is not None:
                break
        vals['inv'].append(inv_error)
        # vals['invf'].append(np.linalg.norm(inv_theta - fwd_theta_norm) / np.linalg.norm(fwd_theta_norm))
        print(theta_norm.flatten())
        print(fwd_theta_norm.flatten())
        # print(inv_theta.flatten())
            # except RuntimeError:
                # vals['inv'].append(float('nan'))

        # for i in range (3, max_l):
        rand_theta = generate_random_vec(dim = theta.shape[1]).reshape([1, theta.shape[1]])
        vals['rand'].append(np.linalg.norm(rand_theta-theta) / np.linalg.norm(theta))
    with open(os.path.join(working_directory, 'results', 'phased-elim-amzn'), 'wb') as file:
        pickle.dump(vals, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(vals)
    return vals


def run_movielens_test():
    import sys
    import os
    sys.path.append(os.path.abspath(working_directory))
    from sampler import SyntheticSampler
    from test_util import test_Phased_Elim
    from alg_util import generate_random_vec
    from estimators import estimate_phased_elim_means, estimate_phased_elim_means_slow
    import numpy as np
    vals = {'fwd': [], 'invf': [], 'inv': [], 'rand': []}
    ord = 2
    for j in range(1):
        # sampler = SyntheticSampler(2000, dim=10, ord=ord)
        # theta, action_set, sigma = sampler.sample()
        # KEEP BELOW
        user_factors = np.load('data/ml-25m/user_factors.npy')
        item_factors = np.load('data/ml-25m/item_factors.npy')
        user_factors = user_factors[user_factors.any(axis=1)]
        item_factors = item_factors[item_factors.any(axis=1)]
        choice = np.random.randint(0, user_factors.shape[0])
        movie_choice = np.random.randint(0, item_factors.shape[0], size=8000)
        action_set = item_factors[movie_choice]
        theta = user_factors[choice:choice+1]
        print(theta)
        # KEEP ABOVE
        sigma = 0.2
        # sampler = SyntheticSampler(2000, dim=d, ord=ord)
        # theta, action_set, sigma = sampler.sample()
        R = np.max(action_set @ theta.T)
        a_star = action_set[np.argmax(action_set @ theta.T)]#[:, np.newaxis]
        # print(a_star.shape)
        # print(R)
        max_l = 5
        fwd_theta, _, alg, max_l = test_Phased_Elim(theta, action_set, sigma, max_l=max_l, iota=0.8)
        # for i in range(, max_l):
        fwd_theta = alg.theta_hats[-1].T
        fwd_theta_norm = fwd_theta / np.linalg.norm(fwd_theta, ord=1)
        theta_norm = theta / np.linalg.norm(theta, ord=1)
        vals['fwd'].append(np.linalg.norm(fwd_theta_norm - theta_norm) / np.linalg.norm(theta_norm))

        # for i in range (3, max_l):
            # try:
        inv_error, inv_theta = estimate_phased_elim_means_slow(alg, a_star, R, theta, l=max_l - 2)
        vals['inv'].append(inv_error)
        vals['invf'].append(np.linalg.norm(inv_theta - fwd_theta_norm) / np.linalg.norm(fwd_theta_norm))
        print(theta_norm.flatten())
        print(fwd_theta_norm.flatten())
        print(inv_theta.flatten())
            # except RuntimeError:
                # vals['inv'].append(float('nan'))

        # for i in range (3, max_l):
        rand_theta = generate_random_vec(dim = theta.shape[1]).reshape([1, theta.shape[1]])
        vals['rand'].append(np.linalg.norm(rand_theta-theta) / np.linalg.norm(theta))
    with open(os.path.join(working_directory, 'results', 'phased-elim-movielens'), 'wb') as file:
        pickle.dump(vals, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(vals)
    return vals


@python_app
# use runsynthetic instead
def run_test_per_dim(d = 1, working_dir=working_directory):
    import sys
    import os
    sys.path.append(os.path.abspath(working_directory))
    from sampler import SyntheticSampler
    from test_util import test_Phased_Elim
    from alg_util import generate_random_vec
    from estimators import estimate_phased_elim_means, estimate_phased_elim_means_slow
    import numpy as np
    vals = {'fwd': [], 'inv': [], 'rand': []}
    ord = 2
    for j in range(10):
        sampler = SyntheticSampler(2000, dim=d, ord=ord)
        theta, action_set, sigma = sampler.sample()
        print(theta.shape)
        print(action_set.shape)
        R = np.max(action_set @ theta.T)
        max_l = 8
        fwd_theta, _, alg = test_Phased_Elim(theta, action_set, sigma, max_l=max_l)
        for i in range(4, max_l):
            vals['fwd'].append(np.linalg.norm(alg.theta_hats[i].T-theta) / np.linalg.norm(theta))

        for i in range (4, max_l):
            try:
                inv_error, inv_theta = estimate_phased_elim_means(alg, R, theta, l=i)
                vals['inv'].append(inv_error)
            except RuntimeError:
                vals['inv'].append(float('nan'))

        for i in range (4, max_l):
            rand_theta = generate_random_vec(dim = theta.shape[1]).reshape([1, theta.shape[1]])
            vals['rand'].append(np.linalg.norm(rand_theta-theta) / np.linalg.norm(theta))
    #with open(os.path.join(working_directory, 'results', 'phased-elim-l%d-d=%d' % (ord, ord, d)), 'wb') as file:
    #    pickle.dump(vals, file, protocol=pickle.HIGHEST_PROTOCOL)
    return vals


@python_app
# use runsynthetic instead
def run_test_per_arms(arms = 2048, d = 1):
    import sys
    import os
    from sampler import SyntheticSampler
    from test_util import test_Phased_Elim
    from alg_util import generate_random_vec
    from estimators import estimate_phased_elim_means
    import numpy as np
    vals = {'fwd': [], 'inv': [], 'rand': []}
    ord = 5
    for j in range(10):
        sampler = SyntheticSampler(arms, dim=d, ord=ord)
        theta, action_set, sigma = sampler.sample()
        R = np.max(action_set @ theta.T)
        max_l = 8
        fwd_theta, _, alg = test_Phased_Elim(theta, action_set, sigma, max_l=max_l)
        for i in range(4, max_l):
            vals['fwd'].append(np.linalg.norm(alg.theta_hats[i].T-theta) / np.linalg.norm(theta))

        for i in range (4, max_l):
            try:
                inv_error, inv_theta = estimate_phased_elim_means(alg, R, theta, l=i)
                vals['inv'].append(inv_error)
            except RuntimeError:
                vals['inv'].append(float('nan'))

        for i in range (4, max_l):
            rand_theta = generate_random_vec(dim = theta.shape[1]).reshape([1, theta.shape[1]])
            vals['rand'].append(np.linalg.norm(rand_theta-theta) / np.linalg.norm(theta))
    with open('/home/jim/narms/phased-elim-l%d-arms=%d' % (ord, arms), 'wb') as file:
        pickle.dump(vals, file, protocol=pickle.HIGHEST_PROTOCOL)
    return vals

@python_app
def run_synthetic(theta=None, action_set=None, sigma=None, d = 1, ord=2, working_dir=working_directory):
    import sys
    import os
    sys.path.append(os.path.abspath(working_dir))
    from sampler import SyntheticSampler
    from test_util import test_Phased_Elim
    from alg_util import generate_random_vec
    from estimators import estimate_phased_elim_means, estimate_phased_elim_subopt_gap, estimate_phased_elim_means_slow
    import numpy as np
    def compute_gamma(action_set):
        action_set = action_set / np.linalg.norm(action_set, axis=-1, keepdims=True)
        cos_sim = action_set @ action_set.T
        np.fill_diagonal(cos_sim, -1)
        best_angles = np.min(np.arccos(np.clip(cos_sim, a_min=-1, a_max=1)), axis=-1)
        gamma = np.max(best_angles)
        return gamma
    vals = {'fwd': [], 'inv': [], 'invf': [], 'rand': []}
    # for j in range(num_trials):
    if theta is None or action_set is None or sigma is None:
        sampler = SyntheticSampler(4000, dim=d, ord=ord)
        theta, action_set, sigma = sampler.sample()
    gamma = compute_gamma(action_set)
    print(theta.shape)
    # print(action_set.shape)
    R = np.max(action_set @ theta.T)
    a_star = action_set[np.argmax(action_set @ theta.T)]#[:, np.newaxis]
    print(a_star.shape)
    max_l = 6
    fwd_theta, _, alg, max_l = test_Phased_Elim(theta, action_set, sigma, max_l=max_l, iota=0.8)
    for i in range(1, max_l):
        fwd_theta = alg.theta_hats[-1].T
        fwd_theta_norm = fwd_theta / np.linalg.norm(fwd_theta, ord=1)
        theta_norm = theta / np.linalg.norm(theta, ord=1)
        vals['fwd'].append(np.linalg.norm(fwd_theta_norm - theta_norm) / np.linalg.norm(theta_norm))

    for i in range (1, max_l):
        inv_error, inv_theta = estimate_phased_elim_means_slow(alg, a_star, R, theta, l=i)
        fwd_theta = alg.theta_hats[1].T
        fwd_theta_norm = fwd_theta / np.linalg.norm(fwd_theta, ord=1)
        if inv_error is None or fwd_theta_norm is None or inv_theta is None:
            invf_error = None
        else:
            invf_error = np.linalg.norm(inv_theta - fwd_theta_norm) / np.linalg.norm(fwd_theta_norm)
        vals['invf'].append(invf_error)
        vals['inv'].append((alg.T[i], inv_error, invf_error, i))
        # vals['mu'].append(inv_mu)
        # try:
            # inv_error, inv_theta = estimate_phased_elim_means(alg, R, theta, l=i)
            # vals['inv'].append(inv_error)
        # except RuntimeError:
            # vals['inv'].append(float('nan'))

    for i in range (1, max_l):
        rand_theta = generate_random_vec(dim = theta.shape[1]).reshape([1, theta.shape[1]])
        vals['rand'].append(np.linalg.norm(rand_theta-theta) / np.linalg.norm(theta))
    #with open(os.path.join(working_directory, 'results', 'synthetic-l%d-d=%d' % (ord, d)), 'wb') as file:
        #pickle.dump(vals, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(vals['inv'])
    print(vals['invf'])
    return vals, gamma

# results_dict = {}
# workers = []
# for i in range(6, 15):
    # workers.append(run_test_per_arms(arms=2**i, d=8))
    # # vals = run_test_per_dim(d=i)
# for i, worker in enumerate(workers):
    # results_dict[i] = worker.result()
    # print('Completed exp %d' % (i+8))
# print(results_dict)
# run_test_per_dim(8)
#run_movielens_test()
#exit()

# UNCOMMENT TO RUN AMAZON MUSIC TESTS
# run_amzn_test()
# exit()

# UNCOMMENT TO RUN SYNTHETIC TESTS
ord = 2
workers = []
inv_err = {}
rand_err = {}
fwd_err = {}
# mu_err = {}
ntrials = 1
gammas = {}
d_min = 4
d_max = 5
for d in range(d_min, d_max):
    gammas[d] = []
    # inv_err[d] = {}
    inv_err[d] = []
    rand_err[d] = []
    fwd_err[d] = []
    # mu_err[d] = []
    for num_trial in range(ntrials):
        #sampler = SyntheticSampler(4000, dim=d, ord=ord)
        theta, action_set, sigma = None, None, None#sampler.sample()
        #gamma = compute_gamma(action_set)
        # print(gamma)
        #gammas[d].append(gamma)
        workers.append(run_synthetic(theta, action_set, sigma, d=d, ord=ord, working_dir=working_directory))
    print(d)

for d in tqdm(range(d_min, d_max)):
    for num_trial in range(ntrials):
        result, gamma = workers[ntrials*(d-d_min)+num_trial].result()
        fwd_res = result['fwd']
        rand_res = result['rand']
        # mu_res = result['mu']
        result = result['inv']

        for i, val in enumerate(result):
            inv_err[d].append((val[0], val[1], val[2], val[3]))
            fwd_err[d].append(fwd_res[i])
            rand_err[d].append(rand_res[i])
            # mu_err[d].append(mu_res[i])
        gammas[d].append(gamma)
    # print(inv_err[d])
            # inv_err[d][int(val[0])] = val[1]
        # inv_err[d].append(workers[ntrials*d+num_trial].result()['inv'][-1])

    # inv_err[d] = sum(inv_err[d]) / ntrials

with open(os.path.join(working_directory, 'results', 'subopt-synthetic-multi-l%d' % (ord)), 'wb') as file:
    pickle.dump(inv_err, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(working_directory, 'results', 'subopt-synthetic-multi-l%d-fwd' % (ord)), 'wb') as file:
    pickle.dump(fwd_err, file, protocol=pickle.HIGHEST_PROTOCOL)

# with open(os.path.join(working_directory, 'results', 'subopt-synthetic-multi-l%d-mu' % (ord)), 'wb') as file:
    # pickle.dump(mu_err, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(working_directory, 'results', 'subopt-synthetic-multi-l%d-gamma' % (ord)), 'wb') as file:
    pickle.dump(gamma, file, protocol=pickle.HIGHEST_PROTOCOL)
#with open('phased-elim-l1', 'wb') as file:
#    pickle.dump(results_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

