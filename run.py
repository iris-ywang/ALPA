import pandas as pd
import os
import numpy as np

from pairwise_approach import run_active_learning_pairwise_approach
from pa_basics.import_chembl_data import dataset
from pa_basics.split_data import initial_split_dataset_by_size
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Cu
ML_REG = RandomForestRegressor(random_state=5963, n_jobs=-1)
ML_CLS = RandomForestClassifier(random_state=5963, n_jobs=-1)

CONNECTION = "//input_data//"
DATASET_FILENAME = "data_CHEMBL202.csv"


if __name__ == '__main__':
    # only one of them can be True.
    rank_only = True
    uncertainty_only = False
    ucb = False
    #

    train_test = dataset(os.getcwd() + CONNECTION + DATASET_FILENAME, shuffle_state=1)
    data = initial_split_dataset_by_size(train_test, 50)
    batch_id_record, top_y_record, mse_record = run_active_learning_pairwise_approach(data, ML_REG, ML_CLS, rank_only, uncertainty_only, ucb)


