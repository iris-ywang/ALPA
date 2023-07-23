from copy import deepcopy
import numpy as np
import random
import logging
from time import perf_counter
from itertools import permutations, product
from sklearn.metrics import mean_squared_error

from pa_basics.all_pairs import paired_data_by_pair_id
from pa_basics.split_data import pair_test_with_train
from pa_basics.rating import rating_trueskill
from pa_basics.run_utils import (
    estimate_counting_stats_for_leave_out_test_set,
    build_ml_model,
    calculate_signed_pairwise_differences_from_y,
    check_batch,
    find_top_x,
    find_how_many_of_batch_id_in_top_x_pc
)

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def run_pairwise_approach_training(
        ml_model_reg,
        ml_model_cls,
        all_data: dict,
        batch_size=10000000,
        warm_start_availability=False,
        normal=False):
    """
    all_data needs to include:
        1. train_test: np.array in shape of [y, x1, x2, ...], train and test all together.
        2. train_pair_ids: list of index IDs
        3. c2_test_pair_ids: list of index IDs
        4. c3_test_pair_ids: list of index IDs
        5. train_ids: list of index IDs
        6. test_ids: list of index IDs
    ml_model_reg/ml_model_cls: sklearn model object with parameters set.

    """
    runs_of_estimators = len(all_data["train_pair_ids"]) // batch_size
    # if normal is False, None will be returned; else, make a copy of incoming ml method to avoid cross-changes.
    ml_model_reg_normal = deepcopy(ml_model_reg) if normal else None
    Y_pa_c1 = []

    if runs_of_estimators < 1:
        train_pairs_batch = paired_data_by_pair_id(data=all_data["dataset"],
                                                   pair_ids=all_data['train_pair_ids'])
        Y_pa_c1 += list(train_pairs_batch[:, 0])

        train_pairs_for_sign = np.array(train_pairs_batch)
        train_pairs_for_sign[:, 0] = np.sign(train_pairs_for_sign[:, 0])
        ml_model_cls = build_ml_model(ml_model_cls, train_pairs_for_sign)

        train_pairs_for_abs = np.absolute(train_pairs_batch)
        ml_model_reg_abs = build_ml_model(ml_model_reg, train_pairs_for_abs)

        if normal:
            ml_model_reg_normal = build_ml_model(ml_model_reg, train_pairs_batch)

    else:
        raise NotImplementedError("Size too big. Code not yet completed.")
    return ml_model_reg_abs, ml_model_cls, ml_model_reg_normal, Y_pa_c1


def run_pairwise_approach_testing(
        ml_model_reg_abs,
        ml_model_cls,
        all_data: dict,
        c2_or_c3: str,
        batch_size=1000000,
        sign=True,
        abs=True,
        normal=False,
        ml_model_reg_normal=None):

    if c2_or_c3 == "c2":
        test_pair_ids = all_data["c2_test_pair_ids"]
    elif c2_or_c3 == "c3":
        test_pair_ids = all_data["c3_test_pair_ids"]
    elif c2_or_c3 == "c2_lo":
        test_pair_ids = all_data["left_out_test_c2_pair_ids"]

    number_test_batches = len(test_pair_ids) // batch_size
    if number_test_batches < 1: number_test_batches = 0

    Y_pa_sign, Y_pa_dist, Y_pa_normal = [], [], []
    Y_pa_true = []

    for test_batch in range(number_test_batches + 1):
        if test_batch != number_test_batches:
            test_pair_id_batch = test_pair_ids[
                                 test_batch * batch_size: (test_batch + 1) * batch_size]
        else:
            test_pair_id_batch = test_pair_ids[test_batch * batch_size:]
        start = perf_counter()
        test_pairs_batch = paired_data_by_pair_id(data=all_data["dataset"],
                                                  pair_ids=test_pair_id_batch)
        end = perf_counter()
        logging.info(f"Time required to pair {len(test_pair_id_batch)} pairs is {end - start} s.")
        Y_pa_true += list(test_pairs_batch[:, 0])
        if sign: Y_pa_sign += list(ml_model_cls.predict(test_pairs_batch[:, 1:]))
        if abs: Y_pa_dist += list(ml_model_reg_abs.predict(np.absolute(test_pairs_batch[:, 1:])))
        if normal and (ml_model_reg_normal is not None):
            Y_pa_normal += list(ml_model_reg_normal.predict(test_pairs_batch[:, 1:]))
        if (test_batch + 1) * batch_size >= len(test_pair_ids): break

    return Y_pa_sign, Y_pa_dist, Y_pa_normal, Y_pa_true


def find_next_batch_pairwise_approach(
        all_data: dict,
        Y_pa_c2_sign,
        Y_pa_c2_norm,
        rank_only: bool,
        uncertainty_only: bool,
        ucb: bool,
        batch_size: int,
        ucb_weighting: float = 0.5,
):
    """Only uses c2_test_pairs to rank and estimate uncertainty."""
    # Check that rank_only, uncertainty_only and ucb can only have one True
    if (rank_only and uncertainty_only) or (rank_only and ucb) or (uncertainty_only and ucb):
        raise ValueError("Can only return batch for one type of selection method.")

    if not (rank_only or uncertainty_only or ucb):
        logging.info("NOTE: None of the acquisition func is specified. Running random selection.")
    # in case rank_only == uncertainty_only == ucb == False, just return random pick
    batch_ids = random.sample(all_data["test_ids"], batch_size)

    # returns a ranking of test samples
    y_ranking_all = rating_trueskill(
        Y_pa_c2_sign, all_data["c2_test_pair_ids"], all_data["y_true"]
    )
    y_ranking = y_ranking_all[all_data["test_ids"]]
    y_ranking_normalised = y_ranking / np.linalg.norm(y_ranking)

    y_mean, y_var = estimate_y_from_Yc2(
        Y_pa_c2_norm, all_data["c2_test_pair_ids"], all_data["test_ids"], all_data["y_true"]
    )
    y_var_normalised = y_var / np.linalg.norm(y_var)

    if rank_only:
        batch_ids = find_top_x(x=batch_size, test_ids=all_data["test_ids"], y_test_score=y_ranking_normalised)
    if uncertainty_only:
        batch_ids = find_top_x(x=batch_size, test_ids=all_data["test_ids"], y_test_score=y_var_normalised)
    if ucb:
        ucb = y_ranking_normalised + ucb_weighting * y_var_normalised
        batch_ids = find_top_x(x=batch_size, test_ids=all_data["test_ids"], y_test_score=ucb)

    top_y = find_how_many_of_batch_id_in_top_x_pc(batch_ids, all_data["train_test"][:, 0], 0.1)

    return batch_ids, [top_y]


def metrics_for_leave_out_test_pairwise_approach(all_data: dict, Y_pa_c2_norm_lo, Y_pa_c2_sign_LO):
    y_test_pred, _ = estimate_y_from_Yc2(
        Y_pa_c2_norm_lo,
        all_data["left_out_test_c2_pair_ids"],
        all_data["left_out_test_ids"],
        all_data["y_true"]
    )
    y_test_true = all_data["left_out_test_set"][:, 0]
    model_mse = mean_squared_error(y_test_true, y_test_pred)

    y_ranking_all = rating_trueskill(
        Y_pa_c2_sign_LO, all_data["left_out_test_c2_pair_ids"], all_data["y_true"]
    )

    metrics_stats = estimate_counting_stats_for_leave_out_test_set(
        y_ranking_all[all_data["left_out_test_ids"]],
        y_ranking_all[all_data["train_ids"] + all_data["test_ids"]],
        all_data["y_true"],
        all_data["left_out_test_ids"],
        all_data["train_ids"] + all_data["test_ids"],
        top_pc=0.1
    )
    return [model_mse] + metrics_stats


def estimate_y_from_Yc2(Y_pa_c2, c2_test_pair_ids, test_ids, y_true, Y_weighted=None):
    """
    Estimate activity values from C2-type test pairs via arithmetic mean or weighted average, It is calculated by
    estimating y_test from [Y_(test, train)_pred + y_train_true] and [ - Y_(train, test)_pred + y_train_true]

    :param Y_pa_c2: np.array of (predicted) differences in activities for C2-type test pairsc
    :param c2_test_pair_ids: list of tuples, each specifying samples IDs for a c2-type pair.
            * Y_pa_c2 and c2_test_pair_ids should match in position; their length should be the same.
    :param test_ids: list of int for test sample IDs
    :param y_true: np.array of true activity values of all samples
    :param Y_weighted: np.array of weighting of each Y_pred (for example, from model prediction probability)
    :return: np.array of estimated activity values for test set
    """
    if y_true is None:
        y_true = y_true
    if Y_weighted is None:  # linear arithmetic
        Y_weighted = np.ones((len(Y_pa_c2)))

    records = [[] for _ in range(len(y_true))]

    for pair in range(len(Y_pa_c2)):
        ida, idb = c2_test_pair_ids[pair]
        delta_ab = Y_pa_c2[pair]
        weight = Y_weighted[pair]

        if ida in test_ids:
            # (test, train)
            weighted_estimate = (y_true[idb] + delta_ab) * weight
            records[ida] += [weighted_estimate]

        elif idb in test_ids:
            # (train, test)
            weighted_estimate = (y_true[ida] - delta_ab) * weight
            records[idb] += [weighted_estimate]

    records_test = [x for x in records if x != []]
    assert len(records_test) == len(test_ids)
    y_estimations_normal = np.array(records_test)  # (test_set_size, n_estimations)
    mean = y_estimations_normal.mean(axis=1)
    var = np.var(y_estimations_normal, axis=1)
    return mean, var


def estimate_Y_from_sign_and_abs(all_data, Y_pa_c2_sign, Y_pa_c2_abs, c2_or_c2_lo="c2"):
    """Returns the combined results same as Y_c2_norm."""
    if c2_or_c2_lo == "c2": test_pair_ids = all_data["c2_test_pair_ids"]
    elif c2_or_c2_lo == "c2_lo": test_pair_ids = all_data["left_out_test_c2_pair_ids"]
    y_ranking_all = rating_trueskill(
        Y_pa_c2_sign, test_pair_ids, all_data["y_true"]
    )
    Y_c2_pred, _ = calculate_signed_pairwise_differences_from_y(test_pair_ids, y_ranking_all)
    Y_c2_pred_sign = np.sign(Y_c2_pred)
    return Y_c2_pred_sign * Y_pa_c2_abs


def find_batch_with_pairwise_approach(all_data: dict, ml_model_reg, ml_model_cls, rank_only, uncertainty_only, ucb, batch_size):
    calculate_normal_reg = False
    # if normal=False, ml_model_reg_normal = None, then Y_pa_c2_norm = []
    logging.info("PA - training...")
    ml_model_reg_abs, ml_model_cls, ml_model_reg_normal, Y_pa_c1 = \
        run_pairwise_approach_training(ml_model_reg, ml_model_cls, all_data,
                                       normal=calculate_normal_reg)

    logging.info("PA - testing...")
    Y_pa_c2_sign, Y_pa_c2_abs, Y_pa_c2_norm, Y_pa_c2_true = \
        run_pairwise_approach_testing(
            ml_model_reg_abs=ml_model_reg_abs,
            ml_model_cls=ml_model_cls,
            all_data=all_data,
            c2_or_c3="c2",
            sign=True,
            abs=True,
            normal=False,
            ml_model_reg_normal=ml_model_reg_normal)

    if not calculate_normal_reg:
        Y_pa_c2_norm = estimate_Y_from_sign_and_abs(all_data, Y_pa_c2_sign, Y_pa_c2_abs)

    logging.info("PA - selecting batch...")
    batch_ids, metrics = find_next_batch_pairwise_approach(
        all_data=all_data,
        Y_pa_c2_sign=Y_pa_c2_sign,
        Y_pa_c2_norm=Y_pa_c2_norm,
        rank_only=rank_only, uncertainty_only=uncertainty_only, ucb=ucb, batch_size=batch_size
    )
    if all_data["left_out_test_ids"] is not None:
        logging.info("Testing on leave-out test set...")
        Y_pa_c2_sign_LO, Y_pa_c2_abs_LO, Y_pa_c2_norm_LO, Y_pa_c2_true_LO = \
            run_pairwise_approach_testing(
                ml_model_reg_abs=ml_model_reg_abs,
                ml_model_cls=ml_model_cls,
                all_data=all_data,
                c2_or_c3="c2_lo",
                sign=True,
                abs=True,
                normal=False,
                ml_model_reg_normal=ml_model_reg_normal)

        if not Y_pa_c2_norm_LO:
            Y_pa_c2_norm_LO = estimate_Y_from_sign_and_abs(all_data, Y_pa_c2_sign_LO, Y_pa_c2_abs_LO, "c2_lo")

        metrics_lo = metrics_for_leave_out_test_pairwise_approach(all_data, Y_pa_c2_norm_LO, Y_pa_c2_sign_LO)
        metrics += metrics_lo

    return batch_ids, metrics


def run_active_learning_pairwise_approach(
        all_data: dict,
        ml_model_reg, ml_model_cls,
        rank_only, uncertainty_only, ucb,
        batch_size=10
    ):
    """all_data should be a starting batch reflected by train_ids and test_ids.
    e,g. len(test_ids) = 50. """

    batch_id_record = []
    metrics_record = []  # record of metrics
    logging.info("Looping ALPA...")
    all_data = dict(all_data)

    for batch_no in range(0, 20):  # if batch_size = 10, loop until train set size = 550.
        logging.info(f"Now running batch number {batch_no}")

        print(f"Size of train, test and c2: "
              f"{len(all_data['train_ids'])}, "
              f"{len(all_data['test_ids'])}, "
              f"{len(all_data['c2_test_pair_ids'])}")

        batch_ids, metrics = find_batch_with_pairwise_approach(
            all_data, ml_model_reg, ml_model_cls,
            rank_only=rank_only, uncertainty_only=uncertainty_only, ucb=ucb, batch_size=batch_size
        )
        batch_id_record.append(batch_ids)
        metrics_record.append(metrics)

        check_batch(batch_ids, all_data["train_ids"])

        train_ids = all_data["train_ids"] + batch_ids
        test_ids = list(set(all_data["test_ids"]) - set(batch_ids))
        left_out_test_ids = all_data["left_out_test_ids"]
        c1_keys_del = list(permutations(train_ids, 2)) + [(a, a) for a in train_ids]
        c2_keys_del = pair_test_with_train(train_ids, test_ids)
        c3_keys_del = list(permutations(test_ids, 2)) + [(a, a) for a in test_ids]
        left_out_c2_pair_ids = pair_test_with_train(train_ids, left_out_test_ids)
        all_data["train_ids"] = train_ids
        all_data["test_ids"] = test_ids
        all_data["train_pair_ids"] = c1_keys_del
        all_data["c2_test_pair_ids"] = c2_keys_del
        all_data["c3_test_pair_ids"] = c3_keys_del
        all_data["left_out_test_c2_pair_ids"] = left_out_c2_pair_ids
        if not test_ids: break

    return batch_id_record, metrics_record
