import os
#import argparse
import pandas as pd
#import utils
#import hyperparameters

class dataread:
    

    def __init__(self, data_dir, train_file, test_file, val_file):
        self.data_dir = data_dir
        self.train_file = train_file
        self.test_file = test_file
        self.val_file = val_file
        self.files = {
            "train": os.path.join(data_dir, self.train_file),
            "test": os.path.join(data_dir, self.test_file),
            "val": os.path.join(data_dir, self.val_file)
        }

    def read_files(self):
        return {key: pd.read_csv(value) for key, value in self.files.items()}

    # def read_files(self):
    #     dfs = {}
    #     for key, file_path in self.files.items():
    #         df = pd.read_csv(file_path, nrows=100)
    #         dfs[key] = df
    #     return dfs

#python <file_name>.py --data_dir <path_to_data_directory> --train_file <train_file_name> --test_file <test_file_name> --val_file <val_file_name>