import numpy as np
import logging
from itertools import permutations, product


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


def calculate_pairwise_differences_from_y(
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
