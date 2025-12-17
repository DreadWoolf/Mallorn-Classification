import numpy as np
import pandas as pd
import os

main_folder = "mallorn-astronomical-classification-challenge"
data_folder = "Data"

path = os.path.join(os.getcwd(), main_folder)
data_path = os.path.join(os.getcwd(), data_folder)

FILTERS = ["u", "g", "r", "i", "z", "y"]

def build_per_filter_features(df: pd.DataFrame, band: str, features: pd.DataFrame) -> pd.DataFrame:
    df_local = df.copy()
    df_local["Time (MJD)"] = pd.to_numeric(df_local["Time (MJD)"], errors="coerce")
    df_local["Flux"] = pd.to_numeric(df_local["Flux"], errors="coerce")
    df_local["Flux_err"] = pd.to_numeric(df_local["Flux_err"], errors="coerce")

    x = df_local[df_local["Filter"] == band].copy()

    # Om bandet saknas helt: lägg till NaN-kolumner (om de inte finns) och returnera
    if x.empty:
        for c in [
            f"n_obs_{band}", f"t_span_{band}", f"mean_flux_{band}", f"std_flux_{band}",
            f"max_flux_{band}", f"min_flux_{band}", f"mean_flux_err_{band}",
            f"mean_snr_{band}", f"t_peak_{band}", f"flux_peak_{band}"
        ]:
            if c not in features.columns:
                features[c] = np.nan
        return features

    agg = (
        x.groupby("object_id", as_index=False)
         .agg(
             n_obs=("Flux", "size"),
             t_first=("Time (MJD)", "min"),
             t_last=("Time (MJD)", "max"),
             mean_flux=("Flux", "mean"),
             std_flux=("Flux", "std"),
             max_flux=("Flux", "max"),
             min_flux=("Flux", "min"),
             mean_flux_err=("Flux_err", "mean"),
         )
    )
    agg[f"t_span_{band}"] = agg["t_last"] - agg["t_first"]

    x["snr"] = x["Flux"] / x["Flux_err"].replace(0, np.nan)
    snr = (
        x.groupby("object_id", as_index=False)["snr"]
         .mean()
         .rename(columns={"snr": f"mean_snr_{band}"})
    )

    x_nonan = x.dropna(subset=["Flux"])
    if x_nonan.empty:
        peak = agg[["object_id"]].copy()
        peak[f"t_peak_{band}"] = np.nan
        peak[f"flux_peak_{band}"] = np.nan
    else:
        peak_idx = x_nonan.groupby("object_id")["Flux"].idxmax()
        peak = (
            x_nonan.loc[peak_idx, ["object_id", "Time (MJD)", "Flux"]]
                  .rename(columns={"Time (MJD)": f"t_peak_{band}", "Flux": f"flux_peak_{band}"})
        )

    agg = agg.rename(columns={
        "n_obs": f"n_obs_{band}",
        "mean_flux": f"mean_flux_{band}",
        "std_flux": f"std_flux_{band}",
        "max_flux": f"max_flux_{band}",
        "min_flux": f"min_flux_{band}",
        "mean_flux_err": f"mean_flux_err_{band}",
    }).drop(columns=["t_first", "t_last"], errors="ignore")

    band_feats = (
        agg.merge(snr, on="object_id", how="left")
           .merge(peak, on="object_id", how="left")
    )

    return features.merge(band_feats, on="object_id", how="left")





def merge_splits(train_or_test = 'train'):

    # Get all unique directories
    data_paths_list = [os.path.join(main_folder, d) for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]


    dfs = []

    for split_path in data_paths_list:
        split_lc = pd.read_csv(
            os.path.join(split_path, f"{train_or_test}_full_lightcurves.csv"),
            sep=","
        )

        features = pd.DataFrame({"object_id": split_lc["object_id"].unique()})

        # Build features for all bands
        for b in FILTERS:
            features = build_per_filter_features(split_lc, b, features)


        dfs.append(features)

    return pd.concat(dfs, ignore_index=True)



def merge_and_save_data():
    os.makedirs(data_folder, exist_ok=True)
    df_dict = {
        "train": pd.DataFrame(),
        "test": pd.DataFrame()
    }
    

    for part in df_dict.keys():
        log_df = pd.read_csv(os.path.join(main_folder, f"{part}_log.csv"), sep=",")

        features = merge_splits(part)

        df_dict[part] = features.merge(
            log_df,
            on=["object_id"],
            how="left"
        )
    
        df_dict[part].to_csv(os.path.join(data_folder, f"MALLORN-data_{part}.csv"), index=False)
    
    return df_dict["train"], df_dict["test"]

