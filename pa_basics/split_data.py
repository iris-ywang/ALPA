import numpy as np
from sklearn.model_selection import KFold
from itertools import permutations, product


def data_check(train_test):
    """
    Check if a dataset has too many repeated activities values. For 5 fold CV, a repeat rate of 15% is used.
    For time-saving experiments, small datasets are used. The range of size of datasets is specified here.
    :param train_test: np.ndarray of filtered dataset - [y, x1, x2, ..., xn]
    :return: bool (True means the dataset is OK for experiment, and vice versa)
    """
    sample_size = np.shape(train_test)[0]
    if sample_size > 100 or sample_size < 90: return False

    my_dict = {i: list(train_test[:, 0]).count(i) for i in list(train_test[:, 0])}
    max_repetition = max(my_dict.values())
    if max_repetition > 0.15 * sample_size: return False

    return True


def get_repetition_rate(train_test) -> float:
    sample_size = np.shape(train_test)[0]
    my_dict = {i: list(train_test[:, 0]).count(i) for i in list(train_test[:, 0])}
    max_repetition = max(my_dict.values())
    return max_repetition / sample_size


def pair_test_with_train(train_ids, test_ids):
    """
    Generate C2-type pairs (test samples pairing with train samples)
    :param train_ids: list of int for training sample IDs
    :param test_ids: list of int for test sample IDs
    :return: list of tuples of sample IDs
    """
    c2test_combs = []
    for comb in product(test_ids, train_ids):
        c2test_combs.append(comb)
        c2test_combs.append(comb[::-1])
    return c2test_combs


def initial_split_dataset_by_size(train_test_all, n_train, proportion_left_out_test=0.0):
    """
    Generate training sets and test sets for standard approach(regression on FP and activities) and for pairwise approach
     (regression on pairwise features and differences in activities) for designated train set size.
     The remaining is the test set.
    :param train_test: np.array of filtered dataset - [y, x1, x2, ..., xn]
    :return: a dict, keys =  fold number, values = the corresponding pre-processed training and test data and
             sample information
    """
    length_left_out_test_set = int(proportion_left_out_test * len(train_test_all))
    length_train_test = len(train_test_all) - length_left_out_test_set
    if length_left_out_test_set > 0:
        train_test = np.array(train_test_all)[:length_train_test]  # first part is for search space
        left_out_test_set = np.array(train_test_all)[length_train_test:]  # second part is for test space
        left_out_test_ids = list(range(length_train_test, len(train_test_all)))

    elif length_left_out_test_set == 0:
        train_test = np.array(train_test_all)
        left_out_test_set = None
        left_out_test_ids = None
        left_out_test_c2_pair_ids = None

    y_true = np.array(train_test_all[:, 0])
    train_ids = list(range(0, n_train))
    test_ids = list(range(n_train, len(train_test)))

    c1_keys_del = list(permutations(train_ids, 2)) + [(a, a) for a in train_ids]
    c2_keys_del = pair_test_with_train(train_ids, test_ids)
    c3_keys_del = list(permutations(test_ids, 2)) + [(a, a) for a in test_ids]

    if left_out_test_set is not None:
        left_out_test_c2_pair_ids = pair_test_with_train(train_ids, left_out_test_ids)

    train_test_data = {'train_test': train_test,
                       "dataset": train_test_all,
                        'train_ids': train_ids, 'test_ids': test_ids,
                        'y_true': y_true, "train_pair_ids": c1_keys_del,
                        "c2_test_pair_ids": c2_keys_del, "c3_test_pair_ids": c3_keys_del,
                        "left_out_test_set": left_out_test_set, "left_out_test_ids": left_out_test_ids,
                        "left_out_test_c2_pair_ids": left_out_test_c2_pair_ids}

    return train_test_data
