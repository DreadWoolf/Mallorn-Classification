

import numpy as np
import pandas as pd
import os



def merge_splits():
    main_folder = "mallorn-astronomical-classification-challenge"
    data_folder = "Data"
    path = os.path.join(os.getcwd(), main_folder)
    data_path = os.path.join(os.getcwd(), data_folder)
    # Path to your main folder

    # Get all unique directories
    data_paths_list = [os.path.join(main_folder, d) for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]

    data_splits = pd.DataFrame()

    dfs = []

    for split in data_paths_list:
        df = pd.read_csv(
            os.path.join(split, "train_full_lightcurves.csv"),
            sep=","
        )
        dfs.append(df)

    df_splits = pd.concat(dfs, ignore_index=True)

    return df_splits

