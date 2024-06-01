"""
An example based off the MovieLens 20M dataset.
This code will automatically download a HDF5 version of this
dataset when first run. The original dataset can be found here:
https://grouplens.org/datasets/movielens/.
Since this dataset contains explicit 5-star ratings, the ratings are
filtered down to positive reviews (4+ stars) to construct an implicit
dataset
"""

from __future__ import print_function

import argparse
import codecs
import logging
import time

import numpy as np
import tqdm

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.movielens import generate_dataset, get_movielens
from implicit.datasets._download import LOCAL_CACHE_DIR
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (
    BM25Recommender,
    CosineRecommender,
    TFIDFRecommender,
    bm25_weight,
)

from eals import ElementwiseAlternatingLeastSquares, load_model

log = logging.getLogger("implicit")


def calculate_similar_movies(output_filename, model_name="als", min_rating=4.0, variant="20m"):
    # read in the input data file
    start = time.time()
    generate_dataset('data/ml-25m', variant='25m', outputpath=LOCAL_CACHE_DIR)
    titles, ratings = get_movielens(variant)

    # remove things < min_rating, and convert to implicit dataset
    # by considering ratings as a binary preference only
    # ratings.data[ratings.data < min_rating] = 0
    # ratings.eliminate_zeros()
    # ratings.data = np.ones(len(ratings.data))
    # print(ratings.data.shape)
    # print(ratings.data[0:70])

    log.info("read data file in %s", time.time() - start)

    # generate a recommender model based off the input params
    if model_name == "als":
        model = AlternatingLeastSquares(factors=10, iterations=20)

        # lets weight these models by bm25weight.
        # log.debug("weighting matrix by bm25_weight")
        # ratings = (bm25_weight(ratings, B=0.9) * 5).tocsr()

    elif model_name == "bpr":
        model = BayesianPersonalizedRanking()

    elif model_name == "lmf":
        model = LogisticMatrixFactorization()

    elif model_name == "tfidf":
        model = TFIDFRecommender()

    elif model_name == "cosine":
        model = CosineRecommender()

    elif model_name == "bm25":
        model = BM25Recommender(B=0.2)

    else:
        raise NotImplementedError(f"model {model_name} isn't implemented for this example")

    user_ratings = ratings.T.tocsr()
    print(user_ratings.shape)

    movie_choice = np.random.randint(0, user_ratings.shape[1], size=8000)
    user_choice = np.random.randint(0, user_ratings.shape[0], size=8000)
    ## COMMENT ME?
    # user_ratings_real = user_ratings[user_choice]
    # user_ratings_real = user_ratings_real[:, movie_choice]
    # user_ratings = user_ratings_real
    ##########
    # print(user_ratings.shape)
    # return
    # print(user_ratings[1, 296])
    # train the model
    # log.debug("training model %s", model_name)
    start = time.time()
    # model.fit(user_ratings)
    model2 = ElementwiseAlternatingLeastSquares(factors=4)
    model2.fit(user_ratings)
    print(model2.user_factors.shape)
    print(model2.item_factors.shape)
    # print(model2.user_factors[1] @ model2.item_factors[[2, 29, 32, 47, 50, 112, 151, 223, 253, 260, 293, 296], :].T)

    # print(model.user_factors.to_numpy().shape)
    # print(model.item_factors.to_numpy().shape)
    # print(model.user_factors.to_numpy()[140])
    # print(model.item_factors.to_numpy()[140])
    # user_factors = model2.user_factors.to_numpy()
    # item_factors = model2.item_factors.to_numpy()
    # print(user_factors[1:2] @ item_factors[[2, 29, 32, 47, 50, 112, 151, 223, 253, 260, 293, 296], :].T)
    # print(item_factors.shape)
    # print(np.max(model.user_factors.to_numpy()[:50000] @ model.item_factors.to_numpy()[:50000].T))
    np.save('data/ml-25m/user_factors.npy', model2.user_factors)
    np.save('data/ml-25m/item_factors.npy', model2.item_factors)
    # log.debug("trained model '%s' in %s", model_name, time.time() - start)
    # log.debug("calculating top movies")

    # user_count = np.ediff1d(ratings.indptr)
    # # print(user_factors[1])
    # # print(model.user_factors.to_numpy()[1:2] @ model.item_factors.to_numpy()[37709:37710].T)
    # # print(user_count)
    # to_generate = sorted(np.arange(len(titles)), key=lambda x: -user_count[x])

    # log.debug("calculating similar movies")
    # with tqdm.tqdm(total=len(to_generate)) as progress:
        # with codecs.open(output_filename, "w", "utf8") as o:
            # for movieid in to_generate:
                # # if this movie has no ratings, skip over (for instance 'Graffiti Bridge' has
                # # no ratings > 4 meaning we've filtered out all data for it.
                # if ratings.indptr[movieid] != ratings.indptr[movieid + 1]:
                    # title = titles[movieid]
                    # for other, score in zip(*model.similar_items(movieid, 11)):
                        # o.write(f"{title}\t{titles[other]}\t{score}\n")
                # progress.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates related movies from the MovieLens 20M "
        "dataset (https://grouplens.org/datasets/movielens/20m/)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="similar-movies.tsv",
        dest="outputfile",
        help="output file name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="als",
        dest="model",
        help="model to calculate (als/bm25/tfidf/cosine)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="20m",
        dest="variant",
        help="Whether to use the 20m, 10m, 1m or 100k movielens dataset",
    )
    parser.add_argument(
        "--min_rating",
        type=float,
        default=4.0,
        dest="min_rating",
        help="Minimum rating to assume that a rating is positive",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    calculate_similar_movies(
        args.outputfile, model_name=args.model, min_rating=args.min_rating, variant=args.variant
    )
