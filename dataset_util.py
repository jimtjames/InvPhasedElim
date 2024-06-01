from typing import Tuple
import numpy as np
from pandas import DataFrame as df
import pandas
from tqdm import tqdm
import torch


def load_battery_dataset() -> Tuple[np.matrix, np.matrix, float]:
    dataset = np.load('battery_hi.npy')
    n_arms = dataset.shape[0]
    max_mean = np.max(dataset[:, 0])
    action_set = np.matrix(np.identity(n_arms))
    # All the sigmas are the same. In the previous work, just divide these by the max mean
    sigma = (dataset[0, 1] / max_mean)
    theta = np.matrix(dataset[:, 0]) / max_mean
    return action_set, theta, sigma

def find_mean(dic):
    all_data = []
    for T in dic.keys():
        for num_arms in dic[T].keys():
            for ele in dic[T][num_arms]:
                    all_data.append(ele)
    return np.mean(all_data)


def load_movielens_dataset():
    ratings = pandas.read_csv('data/ml-25m/ratings.csv')
    # print(np.max(ratings['userId']))
    # users = range(1, 162542)#range(1, 162542)
    # counts = np.zeros(162542)
    # for user in tqdm(users):
        # count = len(ratings[ratings['userId'] == user]['movieId'])
        # counts[user-1] = count
        # # counts.append(count)
        # # scores = ratings[ratings['userId'] == 1]['rating'].to_numpy()
    # best_users = torch.topk(torch.tensor(counts), k=6000)[1]
    # with open('data/ml-25m/top_users.pt', 'wb') as f:
        # torch.save(best_users, f)
    # print(best_users)
    with open('data/ml-25m/top_users.pt', 'rb') as f:
        best_users = torch.load(f)
    ranked_movies = set()
    for user in tqdm(best_users):
        ranked_movies = ranked_movies.union(set(ratings[ratings['userId'] == int(user+1)]['movieId'].to_numpy()))
    ranked_movies = np.array(list(ranked_movies))
    counts = np.zeros_like(ranked_movies)
    best_user_list = (best_users + 1).tolist()
    for i, movie in enumerate(tqdm(ranked_movies)):
        # print(len(set(ratings[ratings['movieId'] == movie]['userId'].to_numpy())))
        count = len(set(ratings[ratings['movieId'] == movie]['userId'].to_numpy()).intersection(set(best_user_list)))
        counts[i] = count
    counts = torch.tensor(counts)
    top, indices = torch.topk(counts, k=4000)
    with open('data/ml-25m/top_movies.pt', 'wb') as f:
        torch.save(indices, f)
    with open('data/ml-25m/top_movies.pt', 'rb') as f:
        best_movies = torch.load(f)
    print(top)
    print(torch.topk(counts, k=10))

    # for user in best_users:
        # user_ranked = set(ratings[ratings['userId'] == int(user)+1]['movieId'].to_numpy())
        # ranked_movies.intersection_update(user_ranked)
        # print(len(ranked_movies))

    # for user in best_users:
        # ratings[ratings['userId'] == user]['movieId']
    # print(ratings)

# load_movielens_dataset()
