# Code for "One Shot Inverse Reinforcement Learning for Stochastic Linear Bandits"

## Environment Setup

Build the environment with:

```
conda env create -f environment.yml
```

This will assumes you have conda installed, and will generate an environment named `invphasedelim`.

## Configuring Experiments

Experiments are configured via `test_phased_elim.py`. By default, the code will run 100 trials of phased elimination on the L2 ball (configured via `opt`) with dimension varying from 3 to 8 via several trials of `run_synthetic`. Results are written out as pickle files to the `results` directory.

To test the MovieLens setup, you must first download the dataset from [here](https://grouplens.org/datasets/movielens/) and move the ml-25m directory under `data`. Afterwards, run `movielens.py` to generate a linear action set. Files for an existing action set are included in this example. Finally, configure `test_phased_elim.py` to use `run_movielens_test` instead of `run_synthetic`.

Please note that memory usage grows dramatically with maximum phase. Using more than 7 phases may result in memory issues and/or precision issues, leading to invalid performance measurements.

You may need a valid Gurobi license to speed up G-Optimal design. For best performance, run on multiple cores by changing the Parsl configuration in `test_phased_elim` for your hardware setup.

## Running experiments

Now you can run an experiment via:

```
conda activate invphasedelim
python3 test_phased_elim.py
```

Results will be stored as a pickle file containing the inverse estimator error as a dictionary of arrays, where keys are the dimension.
