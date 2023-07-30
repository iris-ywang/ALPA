import os
from pa_basics.import_chembl_data import dataset
from pa_basics.split_data import initial_split_dataset_by_size
from pa_basics.all_pairs import paired_data_by_pair_id
from sklearn.ensemble import RandomForestRegressor
from time import perf_counter

if __name__ == '__main__':
    dataset_filename = "data_CHEMBL1841.csv"
    dataset_shuffle_state = 1
    CONNECTION = "//input_data//"

    train_test = dataset(os.getcwd() + CONNECTION + dataset_filename, shuffle_state=dataset_shuffle_state)
    data_size = len(train_test) - 1000
    print("Length of data to pair: ", data_size)
    all_data = initial_split_dataset_by_size(train_test, data_size, proportion_left_out_test=0)

    start = perf_counter()
    train_pairs_batch = paired_data_by_pair_id(data=all_data["dataset"],
                                               pair_ids=all_data['train_pair_ids'])
    end = perf_counter()
    print("Time used :", end-start, "s")