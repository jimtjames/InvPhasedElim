import pickle
with open('results/phased-elim-movielens', 'rb') as f:
    results = pickle.load(f)
print(results)
