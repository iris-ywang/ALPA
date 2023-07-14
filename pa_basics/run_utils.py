import numpy as np
import logging
from itertools import permutations, product
from sklearn.metrics import precision_score, recall_score, f1_score


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


def calculate_signed_pairwise_differences_from_y(
        cx_test_pair_ids, y_pred_all, y_true_all=None
):
    Y_true, Y_pred = [], []
    for comb in cx_test_pair_ids:
        a, b = comb
        Y_pred.append(np.sign(y_pred_all[a] - y_pred_all[b]))

        if y_true_all is not None:
            Y_true.append(np.sign(y_true_all[a] - y_true_all[b]))
    return Y_pred, Y_true


def find_top_x(x: int, test_ids, y_test_score: np.array) -> list:
    if len(test_ids) != len(y_test_score):
        raise ValueError(
            f"Test set ({len(test_ids)}) and y_test_scores({len(y_test_score)}) don't have the same size. "
            "Cannot find top batch."
        )

    # a list of the sequence of y_test in the descending order of activity values
    overall_orders = np.argsort(-y_test_score)
    # find the sequence of test_ids
    ranked_test_ids = np.array(test_ids)[overall_orders]
    top_tests_id = ranked_test_ids[0: x]
    return list(top_tests_id)


def check_batch(batch, train_ids):
    for i in batch:
        if i in train_ids:
            raise IndexError("Batch suggested sample from train_set. ")
    logging.info("Pass check_batch.")


def find_how_many_of_batch_id_in_top_x_pc(batch_ids, y_true, top_pc=0.1):
    number_of_tops = int(len(y_true) * top_pc)
    top_y_true_ids = find_top_x(number_of_tops, list(range(len(y_true))), y_true)
    top_x_pc_y = y_true[top_y_true_ids[-1]]

    count = 0
    for i in batch_ids:
        if y_true[i] >= top_x_pc_y:
            count += 1
    return count


def estimate_counting_stats_for_leave_out_test_set(
        y_lo_test_pred, y_true_train_test, y_true_all, leave_out_test_ids, train_test_ids, top_pc=0.1
):

    array_pred_ids_all = np.array([
        (list(y_lo_test_pred) + list(y_true_train_test)),
        leave_out_test_ids + train_test_ids
    ]).T
    top_tests, tests_better_than_top_train = find_top_test_ids(
        array_pred_ids_all, leave_out_test_ids, train_test_ids, top_pc
    )

    array_pred_ids_all_true = np.array([
        list(y_true_all),
        list(range(len(y_true_all)))
    ]).T
    top_tests_true, tests_better_than_top_train_true = find_top_test_ids(
        array_pred_ids_all_true, leave_out_test_ids, train_test_ids, top_pc
    )
    precision_top, recall_top, f1_top = estimate_precision_recall(
        top_tests_true, top_tests, leave_out_test_ids
    )
    precision_better, recall_better, f1_better = estimate_precision_recall(
        top_tests_true, top_tests, leave_out_test_ids
    )
    return [precision_top, recall_top, f1_top, precision_better, recall_better, f1_better]


def find_top_test_ids(array_pred_ids_all, test_ids, train_ids, top_pc=0.1):
    """For a test set and a atrain set, identify the test samples that have
     y_pred value listed in the top e.g. 10% of the dataset(train + test)"""

    # trains == train samples; tests == test samples.
    # array_pred_ids_all is a np.array of [y, id]: [[y1, 1], [y9, 9], [y4, 4], ...]

    y_pred_all = array_pred_ids_all[:, 0]
    overall_orders = np.argsort(-y_pred_all)  # a list of sample IDs in the descending order of activity values
    ordered_pred_id = array_pred_ids_all[overall_orders]  # ordered [y, id]
    ordered_ids = ordered_pred_id[:, 1]  #ordered ids
    top_trains_and_tests = ordered_ids[0: int(top_pc * len(ordered_ids))]  # top pc of ids
    top_tests = [idx for idx in top_trains_and_tests if idx in test_ids]

    # Find the ID of top train sample in the overall_order
    top_train_order_position = 0
    while True:
        if ordered_ids[top_train_order_position] in train_ids: break
        top_train_order_position += 1
    top_train_id = ordered_ids[top_train_order_position]

    tests_better_than_top_train = list(ordered_ids[:top_train_order_position])
    return top_tests, tests_better_than_top_train


def estimate_precision_recall(top_tests_true, top_tests, test_ids):
    test_samples_boolean_true = [0 for _ in range(len(test_ids))]
    for top_test_id_true in top_tests_true:
        position_in_test_ids = int(np.where(test_ids == top_test_id_true)[0])
        test_samples_boolean_true[position_in_test_ids] = 1

    test_samples_boolean_pred = [0 for _ in range(len(test_ids))]
    for top_test_id in top_tests:
        position_in_test_ids = int(np.where(test_ids == top_test_id)[0])
        test_samples_boolean_pred[position_in_test_ids] = 1

    precision = precision_score(test_samples_boolean_true, test_samples_boolean_pred)
    recall = recall_score(test_samples_boolean_true, test_samples_boolean_pred)
    f1 = f1_score(test_samples_boolean_true, test_samples_boolean_pred)

    return precision, recall, f1