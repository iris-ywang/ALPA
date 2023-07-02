import numpy as np
import random
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pa_basics.run_utils import (
    build_ml_model,
    find_top_x,
)


def run_standard_approach(
        ml_model_reg,
        all_data: dict,
):
    train_test = all_data['train_test']
    train_set = train_test[all_data["train_ids"]]
    test_set = train_test[all_data["test_ids"]]
    sa_model, y_SA = build_ml_model(ml_model_reg, train_set, test_set)
    # y_pred_all = np.array(all_data["y_true"])
    # y_pred_all[all_data["test_ids"]] = y_SA
    return sa_model, y_SA


def find_next_batch_standard_approach(
        all_data: dict,
        ml_model_reg_sa,
        y_SA,
        rank_only: bool,
        uncertainty_only: bool,
        ucb: bool,
        batch_size: int,
        ucb_weighting: float = 0.5,
):
    # Check that rank_only, uncertainty_only and ucb can only have one True
    if (rank_only and uncertainty_only) or (rank_only and ucb) or (uncertainty_only and ucb):
        raise ValueError("Can only return batch for one type of selection method.")

    if not (rank_only or uncertainty_only or ucb):
        print("None of the acquisition func is specified. Running random selection.")
    # in case rank_only == uncertainty_only == ucb == False, just return random pick
    batch_ids = random.sample(all_data["test_ids"], batch_size)

    y_ranking_normalised = y_SA / np.linalg.norm(y_SA)

    y_var, y_mean = estimate_variance_from_random_forest(all_data, ml_model_reg_sa)
    var_normal = y_var / np.linalg.norm(y_var)

    if rank_only:
        batch_ids = find_top_x(x=batch_size, test_ids=all_data["test_ids"], y_test_score=y_ranking_normalised)
    if uncertainty_only:
        batch_ids = find_top_x(x=batch_size, test_ids=all_data["test_ids"], y_test_score=var_normal)
    if ucb:
        ucb = y_ranking_normalised + ucb_weighting * var_normal
        batch_ids = find_top_x(x=batch_size, test_ids=all_data["test_ids"], y_test_score=ucb)

    top_y = max(all_data['y_true'][batch_ids])
    model_mse = mean_squared_error(all_data['y_true'][all_data["test_ids"]], y_mean)

    return batch_ids, (top_y, model_mse)


def estimate_variance_from_random_forest(all_data: dict, ml_model_reg_sa) -> np.array:
    if not isinstance(ml_model_reg_sa, RandomForestRegressor):
        raise TypeError("Current standard approach variance estimation can only take Random Forest")

    test_set = all_data["train_test"][all_data["test_ids"]]
    X = test_set[:, 1:]
    estimations = []
    for model in ml_model_reg_sa.estimators_:
        estimations.append(model.predict(X))
        # model.predict(X) is an 1D array of length test_set_size
    estimations = np.array(estimations)  # (n_estimators, test_set_size)
    mean = estimations.mean(axis=0)
    variance = np.var(estimations, axis=0)
    return variance, mean


def find_batch_with_standard_approach(all_data: dict, ml_model_reg, rank_only, uncertainty_only, ucb, batch_size):
    ml_model_reg_sa, y_SA = run_standard_approach(ml_model_reg, all_data)

    batch_ids, metrics = find_next_batch_standard_approach(
        all_data=all_data,
        ml_model_reg_sa=ml_model_reg_sa,
        y_SA=y_SA,
        rank_only=rank_only,
        uncertainty_only=uncertainty_only,
        ucb=ucb,
        batch_size=batch_size,
    )

    return batch_ids, metrics


def run_active_learning_standard_approach(
    all_data: dict,
    ml_model_reg,
    rank_only, uncertainty_only, ucb,
    batch_size=10
):
    batch_id_record = []
    top_y_record = []  # record of exploitative performance
    mse_record = []  # record of exploration performance
    for batch_no in range(0, 50):  # if batch_size = 10, loop until train set size = 550.
        logging.info(f"Now running batch number {batch_no}")
        batch_ids, metrics = find_batch_with_standard_approach(
            all_data, ml_model_reg,
            rank_only=rank_only, uncertainty_only=uncertainty_only, ucb=ucb, batch_size=batch_size
        )
        batch_id_record.append(batch_ids)
        top_y_record.append(metrics[0])
        mse_record.append(metrics[1])

        train_ids = all_data["train_ids"] + batch_ids
        test_ids = list(set(all_data["test_ids"]) - set(batch_ids))
        # pair keys will not be generated for standard approach
        all_data["train_ids"] = train_ids
        all_data["test_ids"] = test_ids

    return batch_id_record, top_y_record, mse_record
