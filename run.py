import pandas as pd
import os
import numpy as np
import logging
import time
import warnings

from pairwise_approach import run_active_learning_pairwise_approach
from standard_approch import run_active_learning_standard_approach
from pa_basics.import_chembl_data import dataset
from pa_basics.split_data import initial_split_dataset_by_size
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# TODO: modify all_data and pairwise settings in a class.

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Cu
RANDOM_FOREST_ARGS = {"random_state": 5963, "n_jobs": -1}
CONNECTION = "//input_data//"
# DATASET_FILENAME = "data_CHEMBL2024.csv"


# if __name__ == '__main__':
def run_single_dataset(dataset_filename, dataset_shuffle_state=None):
    warnings.filterwarnings("ignore")

    # only one of them can be True.
    logging.info(f"Dataset name: {dataset_filename}")
    rank_only = False
    uncertainty_only = True
    ucb = False
    proportion_leave_out_test = 0.15  # 0 - 1
    logging.info(f"Acquisition function setting: "
                 f"rank_only - {rank_only}, "
                 f"uncertainty_only - {uncertainty_only}, "
                 f"ucb - {ucb}"
                 f"proportion of leave_out_test_set - {proportion_leave_out_test}")

    ML_REG = RandomForestRegressor(**RANDOM_FOREST_ARGS)
    ML_CLS = RandomForestClassifier(**RANDOM_FOREST_ARGS)

    if dataset_shuffle_state is None:
        dataset_shuffle_state = 1

    train_test = dataset(os.getcwd() + CONNECTION + dataset_filename, shuffle_state=dataset_shuffle_state)
    size = len(train_test)
    initial_size = int(0.05 * size)
    batch_size = int(0.01 * size) if int(0.01 * size) != 0 else 1
    logging.info(f"Initial train set size: {initial_size}, batch size: {batch_size}")
    data = initial_split_dataset_by_size(train_test, initial_size, proportion_left_out_test=proportion_leave_out_test)

    logging.info("Starting standard approach active learning...")
    batch_id_record_sa, metrics_record_sa = run_active_learning_standard_approach(
        data, ML_REG, rank_only, uncertainty_only, ucb, batch_size=batch_size
    )
    logging.info("Starting pairwise approach active learning...")
    batch_id_record_pa, metrics_record_pa = run_active_learning_pairwise_approach(
        data, ML_REG, ML_CLS, rank_only, uncertainty_only, ucb, batch_size=batch_size
    )

    timestr = time.strftime("%Y%m%d-%H%M%S")

    if data["left_out_test_set"] is None:
        column_names = ["top_y"]
    else:
        column_names = ["top_y", "mse", "p_t", "r_t", "f_t", "p_b", "r_b", "f_b"]

    summary_sa = pd.DataFrame(
        metrics_record_sa, columns=["SA-" + cn for cn in column_names]
    )
    summary_pa = pd.DataFrame(
        metrics_record_pa, columns=["PA-" + cn for cn in column_names]
    )
    summary = pd.concat([summary_sa, summary_pa], axis=1)

    summary.to_csv("results_summary_"+timestr+".csv", index=False)

    batches_sa = pd.DataFrame(
        batch_id_record_sa
    )
    batches_sa.to_csv("sa_batches"+timestr+".csv", index=False)

    batches_pa = pd.DataFrame(
        batch_id_record_pa
    )
    batches_pa.to_csv("pa_batches"+timestr+".csv", index=False)

    logging.info("End. Results saved.")
    print("\n \n")
    return timestr

# TODO: add saving log file
