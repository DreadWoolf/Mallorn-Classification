

import numpy as np
import pandas as pd
import os


main_folder = "mallorn-astronomical-classification-challenge"
data_folder = "Data"

path = os.path.join(os.getcwd(), main_folder)
data_path = os.path.join(os.getcwd(), data_folder)



def merge_splits(train_or_test = 'train'):

    # Get all unique directories
    data_paths_list = [os.path.join(main_folder, d) for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]


    dfs = []

    for split_path in data_paths_list:
        split = pd.read_csv(
            os.path.join(split_path, f"{train_or_test}_full_lightcurves.csv"),
            sep=","
        )

        dfs.append(build_per_filter_features(split, "?"))

    df_splits = pd.concat(dfs, ignore_index=True)

    return df_splits




def merge_and_save_data():

    df_dict = {
        "train": pd.DataFrame(),
        "test": pd.DataFrame()
    }
    

    for part in df_dict.keys(): #["train", "test"]:
        df = pd.read_csv(os.path.join(main_folder, f"{part}_log.csv"), sep=",")

        features = merge_splits(part)

        df_dict[part] = features.merge(
            df,
            # on=["object_id", "split"],  # or just object_id if unique
            on=["object_id"],  # or just object_id if unique
            how="left"
        )
    
        df_dict[part].to_csv(os.path.join(data_folder, f"MALLORN-data_{part}.csv"), index=False)

    
    return df_dict["train"], df_dict["test"]



def build_per_filter_features(df, string):

    return df
    pass
