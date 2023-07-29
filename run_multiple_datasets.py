import os
import logging
from run import run_single_dataset

if __name__ == '__main__':

    list_of_dataset = os.listdir(os.getcwd() + "//input_data//")
    record = {}
    for dataset_filename in list_of_dataset:
        if "data_CHEMBL" not in dataset_filename:
            continue

        record[dataset_filename] = []
        for random_state in [1]:
            time_str = run_single_dataset(dataset_filename, dataset_shuffle_state=random_state)
            record[dataset_filename].append(time_str)
            print(record)
        logging.info(f"Results dict: {record}")
