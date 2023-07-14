import os
from run import run_single_dataset

if __name__ == '__main__':

    list_of_dataset = os.listdir(os.getcwd() + "//input_data//")
    record = {}
    for dataset_filename in list_of_dataset:
        record[dataset_filename] = []
        for random_state in [1,2,3]:
            time_str = run_single_dataset(dataset_filename, dataset_shuffle_state=random_state)
            record[dataset_filename].append(time_str)
            print(record)