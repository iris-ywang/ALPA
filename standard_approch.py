import numpy as np
import random
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pa_basics.run_utils import (
    estimate_counting_stats_for_leave_out_test_set,
    build_ml_model,
    check_batch,
    find_top_x,
    find_how_many_of_batch_id_in_top_x_pc
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

    metrics = []
    top_y = find_how_many_of_batch_id_in_top_x_pc(batch_ids, all_data["train_test"][:, 0], 0.1)
    metrics.append(top_y)

    if all_data["left_out_test_ids"] is not None:
        metrics_lo = metrics_for_leave_out_test_standard_approach(all_data, ml_model_reg_sa)
        metrics += metrics_lo

    return batch_ids, metrics


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


def metrics_for_leave_out_test_standard_approach(all_data: dict, ml_model_reg):
    x_test = all_data["left_out_test_set"][:, 1:]
    y_test = all_data["left_out_test_set"][:, 0]
    y_pred = ml_model_reg.predict(x_test)
    model_mse = mean_squared_error(y_test, y_pred)

    metrics_stats = estimate_counting_stats_for_leave_out_test_set(
        y_pred,
        all_data["train_test"][:, 0],
        all_data["y_true"],
        all_data["left_out_test_ids"],
        all_data["train_ids"] + all_data["test_ids"],
        top_pc=0.1
    )
    return [model_mse] + metrics_stats


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
    metrics_record = []  # record of metrics
    logging.info("Looping ALSA...")
    all_data = dict(all_data)

    for batch_no in range(0, 20):  # if batch_size = 10, loop until train set size = 550.
        logging.info(f"Now running batch number {batch_no}")
        print(f"Size of train, test and c2: "
              f"{len(all_data['train_ids'])}, "
              f"{len(all_data['test_ids'])}, "
              f"{len(all_data['c2_test_pair_ids'])}")
        batch_ids, metrics = find_batch_with_standard_approach(
            all_data, ml_model_reg,
            rank_only=rank_only, uncertainty_only=uncertainty_only, ucb=ucb, batch_size=batch_size
        )
        batch_id_record.append(batch_ids)
        metrics_record.append(metrics)

        check_batch(batch_ids, all_data["train_ids"])

        train_ids = all_data["train_ids"] + batch_ids
        test_ids = list(set(all_data["test_ids"]) - set(batch_ids))
        # pair keys will not be generated for standard approach
        all_data["train_ids"] = train_ids
        all_data["test_ids"] = test_ids
        if not test_ids: break

    return batch_id_record, metrics_record
