import numpy as np
from itertools import permutations, product

from pa_basics.all_pairs import paired_data_by_pair_id
from pa_basics.rating import rating_trueskill
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

ML_REG = RandomForestRegressor(random_state=5963, n_jobs=-1)
ML_CLS = RandomForestClassifier(random_state=5963, n_jobs=-1)

def build_ml_model(model, train_data, test_data=None):
    """Given ML model(in sklearn format), and train data in the
     shape of [y, x1, x2, ...], fit the model and return model.
     If test data is given, a second parameter is returned. """
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    fitted_model = model.fit(x_train, y_train)

    if type(test_data) == np.ndarray:
        x_test = test_data[:, 1:]
        y_test_pred = fitted_model.predict(x_test)
        return fitted_model, y_test_pred
    else:
        return fitted_model



def run_pairwise_approach_training(
        ml_model_reg,
        ml_model_cls,
        all_data: dict,
        batch_size=1000000,
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
    ml_model_reg_normal = None # if normal is False, None will be returned.
    Y_pa_c1 = []

    if runs_of_estimators < 1:
        train_pairs_batch = paired_data_by_pair_id(data=all_data["train_test"],
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
        # if not warm_start_availability:
        #     print("Warm start not available. Please reduce train set size.")
        #     return (None, None, None, None)

        # for run in range(runs_of_estimators + 1):
        #     if run < runs_of_estimators:
        #         train_ids_per_batch = all_data["train_pair_ids"][run * batch_size:(run + 1) * batch_size]
        #
        #     else:
        #         train_ids_per_batch = all_data["train_pair_ids"][run * batch_size:]
        #
        #     train_pairs_batch = paired_data_by_pair_id(data=all_data["train_test"],
        #                                                     pair_ids=train_ids_per_batch)
        #     Y_pa_c1 += list(train_pairs_batch[:, 0])
        #
        #     train_pairs_for_sign = np.array(train_pairs_batch)
        #     train_pairs_for_sign[:, 0] = np.sign(train_pairs_for_sign[:, 0])
        #     ml_model_cls = build_ml_model(ml_model_cls, train_pairs_for_sign)
        #
        #     train_pairs_for_abs = np.absolute(train_pairs_batch)
        #     ml_model_reg_abs = build_ml_model(ml_model_reg, train_pairs_for_abs)
        #     ml_model_cls.n_estimators += 100
        #     ml_model_reg_abs.n_estimators += 100
        #
        #     if normal:
        #         ml_model_reg_normal = build_ml_model(ml_model_reg, train_pairs_batch)
        #         ml_model_reg_normal.n_estimators += 100

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
        test_pairs_batch = paired_data_by_pair_id(data=all_data["train_test"],
                                                       pair_ids=test_pair_id_batch)
        Y_pa_true += list(test_pairs_batch[:, 0])
        if sign: Y_pa_sign += list(ml_model_cls.predict(test_pairs_batch[:, 1:]))
        if abs: Y_pa_dist += list(ml_model_reg_abs.predict(np.absolute(test_pairs_batch[:, 1:])))
        if normal and ml_model_reg_normal is not None:
            Y_pa_normal += list(ml_model_reg_normal.predict(test_pairs_batch[:, 1:]))
        if Y_pa_dist(test_batch + 1) * batch_size >= len(test_pair_ids): break

    return Y_pa_sign, Y_pa_dist, Y_pa_normal, Y_pa_true


def find_next_batch_pairwise_approach(
        all_data: dict,
        Y_pa_c2_sign,
        Y_pa_c2_norm,
        rank_only: bool,
        uncertainty_only: bool,
        ucb: bool,
        ucb_weighting: float = 0.5,
        batch_size: int=10,
) -> list:
    """Only uses c2_test_pairs to rank and estimate uncertainty."""
    if rank_only or ucb:
        y_ranking = rating_trueskill(
            Y_pa_c2_sign, all_data["c2_test_pair_ids"], all_data["y_true"]
        )[all_data["test_ids"]]
        y_ranking_normalised = y_ranking / np.linalg.norm(y_ranking)

    if uncertainty_only or ucb:
        y_estimations_normal = estimate_y_from_averaging(Y_pa_c2_norm, all_data["c2_test_pair_ids"], all_data["test_ids"], all_data["y_true"])
        sigma_normal = np.var(y_estimations_normal, axis=0)
        sigma_normal = sigma_normal / np.linalg.norm(sigma_normal)

    if rank_only:
        batch_ids = find_top_x(x=batch_size, y_test_score=y_ranking_normalised)
    if uncertainty_only:
        batch_ids = find_top_x(x=batch_size, y_test_score=sigma_normal)
    if ucb:
        ucb = y_ranking_normalised + ucb_weighting * sigma_normal
        batch_ids = find_top_x(x=batch_size, y_test_score=ucb)

    return batch_ids


def estimate_y_from_averaging(Y_pa_c2, c2_test_pair_ids, test_ids, y_true, Y_weighted=None):
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

    records = np.zeros((len(y_true)))
    weights = np.zeros((len(y_true)))

    for pair in range(len(Y_pa_c2)):
        ida, idb = c2_test_pair_ids[pair]
        delta_ab = Y_pa_c2[pair]
        weight = Y_weighted[pair]

        if ida in test_ids:
            # (test, train)
            weighted_estimate = (y_true[idb] + delta_ab) * weight
            records[ida] += weighted_estimate
            weights[ida] += weight

        elif idb in test_ids:
            # (train, test)
            weighted_estimate = (y_true[ida] - delta_ab) * weight
            records[idb] += weighted_estimate
            weights[idb] += weight

    return np.divide(records[test_ids], weights[test_ids])


def find_top_x(x: int, y_test_score: np.array) -> list:
    overall_orders = np.argsort(-y_test_score)  # a list of sample IDs in the descending order of activity values
    top_tests_id = overall_orders[0: x]
    return top_tests_id


def estimate_Y_from_sign_and_abs(all_data, Y_pa_c2_sign, Y_pa_c2_abs):
    """Returns the combined results same as Y_c2_norm."""
    y_ranking_all = rating_trueskill(
        Y_pa_c2_sign, all_data["c2_test_pair_ids"], all_data["y_true"]
    )
    Y_c2_pred, _ = calculate_pairwise_differences_from_y(all_data["c2_test_pair_ids"], y_ranking_all)
    Y_c2_pred_sign = np.sign(Y_c2_pred)
    return Y_c2_pred_sign * Y_pa_c2_abs


def calculate_pairwise_differences_from_y(
        cx_test_pair_ids, y_pred_all, y_true_all=None
):
    Y_true, Y_pred = [], []
    for comb in cx_test_pair_ids:
        a, b = comb
        Y_pred.append(np.sign(y_pred_all[a] - y_pred_all[b]))

        if y_true_all is not None:
            Y_true.append(np.sign(y_true_all[a] - y_true_all[b]))
    return Y_true, Y_pred

def find_batch_with_pairwise_approach(all_data: dict, rank_only, uncertainty_only, ucb):
    ml_model_reg_abs, ml_model_cls, ml_model_reg_normal, Y_pa_c1 = run_pairwise_approach_training(ML_REG,ML_CLS,all_data, normal = False)
    Y_pa_c2_sign, Y_pa_c2_abs, Y_pa_c2_norm, Y_pa_c2_true = run_pairwise_approach_testing(ml_model_reg_abs, ml_model_cls, all_data, "c2")
    batch_ids = find_next_batch_pairwise_approach(all_data, Y_pa_c2_sign, Y_pa_c2_norm, rank_only=rank_only, uncertainty_only=uncertainty_only, ucb=ucb, batch_size=10)

    return batch_ids

def run_active_learning_pairwise_approach(all_data: dict):
    """all_data should be a starting batch reflected by train_ids and test_ids.
    e,g. len(test_ids) = 50. """

    rank_only = True
    uncertainty_only = False
    ucb = False
    batch_id_record = []
    top_y_record = [] # record of exploitative performance
    mse_record = [] # record of exploration performance
    for batch_no in range(0, 50): # if batch_size = 10, loop until train set size = 550.
        batch_ids = find_batch_with_pairwise_approach(all_data, rank_only=rank_only, uncertainty_only=uncertainty_only, ucb=ucb)
        batch_id_record.append(batch_ids)
        top_y_record.append(max(all_data['y_true'][batch_ids]))

        train_ids = all_data["train_ids"] + batch_ids
        test_ids = list(set(all_data["test_ids"]) - set(batch_ids))
        c1_keys_del = list(permutations(train_ids, 2)) + [(a, a) for a in train_ids]
        c2_keys_del = pair_test_with_train(train_ids, test_ids)
        c3_keys_del = list(permutations(test_ids, 2)) + [(a, a) for a in test_ids]
        all_data["train_ids"] = train_ids
        all_data["test_ids"] = test_ids
        all_data["train_pair_ids"] = c1_keys_del
        all_data["c2_test_pair_ids"] = c2_keys_del
        all_data["c3_test_pair_ids"] = c3_keys_del

    return batch_id_record, top_y_record