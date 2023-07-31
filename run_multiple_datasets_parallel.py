import os
import time
import logging
import multiprocessing
from functools import partial
from run import run_single_dataset


def check_dataset_filename(list_of_datasets):
    return [dataset_filename for dataset_filename in list_of_datasets if "data_CHEMBL" in dataset_filename]


if __name__ == '__main__':

    list_of_dataset = os.listdir(os.getcwd() + "//input_data//")
    list_of_dataset = check_dataset_filename(list_of_dataset)

    timestr_start = time.strftime("%Y%m%d-%H%M%S")

    output_path = os.getcwd() + f"//outputs//output_{timestr_start}//"
    os.mkdir(output_path)
    repeated_output_path = [output_path for i in range(len(list_of_dataset))]
    partial_run_single_dataset = partial(run_single_dataset, output_path=output_path)

    record = {}
    with multiprocessing.Pool(processes=5) as executor:
        time_str_results = executor.map(partial_run_single_dataset, list_of_dataset)
    record = dict(zip(list_of_dataset, time_str_results))
    print(record)
    logging.info(f"Results dict: {record}")
