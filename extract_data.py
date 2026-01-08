import numpy as np
import pandas as pd
import os

main_folder = "mallorn-astronomical-classification-challenge"
data_folder = "Data"

path = os.path.join(os.getcwd(), main_folder)
data_path = os.path.join(os.getcwd(), data_folder)

FILTERS = ["u", "g", "r", "i", "z", "y"]

def build_per_filter_features(df: pd.DataFrame, filter_type: str, features: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-filter aggregated light-curve statistics and merges them into the features DataFrame (one row per object_id)

    All new features (per filter type): n_obs, t_span, mean/std/max/min flux, mean flux_err, mean SNR (Signal-to-Noise Ratio), peak time and peak flux.
    Edge cases:
        - If the filter is missing, the corresponding features are NaN.
        - If Flux is all NaN for an object, peak features become NaN.
        - Flux_err==0 is treated as NaN for SNR to avoid division by zero.
    """

    # Makes columns numeric
    df_local = df.copy()
    df_local["Time (MJD)"] = pd.to_numeric(df_local["Time (MJD)"], errors="coerce")
    df_local["Flux"] = pd.to_numeric(df_local["Flux"], errors="coerce")
    df_local["Flux_err"] = pd.to_numeric(df_local["Flux_err"], errors="coerce")

    x = df_local[df_local["Filter"] == filter_type].copy()

    # if the filter is missing, return empty features (NaN)
    if x.empty:
        for c in [
            f"n_obs_{filter_type}", f"t_span_{filter_type}", f"mean_flux_{filter_type}", f"std_flux_{filter_type}",
            f"max_flux_{filter_type}", f"min_flux_{filter_type}", f"mean_flux_err_{filter_type}",
            f"mean_snr_{filter_type}", f"t_peak_{filter_type}", f"flux_peak_{filter_type}"
        ]:
            if c not in features.columns:
                features[c] = np.nan
        return features

    # Makes basic per-object statistics and time span
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
    agg[f"t_span_{filter_type}"] = agg["t_last"] - agg["t_first"]

    # Mean Signal-to-Noise Ratio per-object
    x["snr"] = x["Flux"] / x["Flux_err"].replace(0, np.nan)
    snr = (
        x.groupby("object_id", as_index=False)["snr"]
         .mean()
         .rename(columns={"snr": f"mean_snr_{filter_type}"})
    )

    # Peak time and flux per-object, ignoring NaNs
    x_nonan = x.dropna(subset=["Flux"])
    if x_nonan.empty:
        peak = agg[["object_id"]].copy()
        peak[f"t_peak_{filter_type}"] = np.nan
        peak[f"flux_peak_{filter_type}"] = np.nan
    else:
        peak_idx = x_nonan.groupby("object_id")["Flux"].idxmax()
        peak = (
            x_nonan.loc[peak_idx, ["object_id", "Time (MJD)", "Flux"]]
                  .rename(columns={"Time (MJD)": f"t_peak_{filter_type}", "Flux": f"flux_peak_{filter_type}"})
        )

    # Merges all per-object statistics into the features DataFrame
    agg = agg.rename(columns={
        "n_obs": f"n_obs_{filter_type}",
        "mean_flux": f"mean_flux_{filter_type}",
        "std_flux": f"std_flux_{filter_type}",
        "max_flux": f"max_flux_{filter_type}",
        "min_flux": f"min_flux_{filter_type}",
        "mean_flux_err": f"mean_flux_err_{filter_type}",
    }).drop(columns=["t_first", "t_last"], errors="ignore")

    band_feats = (
        agg.merge(snr, on="object_id", how="left")
           .merge(peak, on="object_id", how="left")
    )

    return features.merge(band_feats, on="object_id", how="left")




def merge_splits(train_or_test = 'train'):

    data_paths_list = [os.path.join(main_folder, d) for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]


    dfs = []

    for split_path in data_paths_list:
        split_lc = pd.read_csv(
            os.path.join(split_path, f"{train_or_test}_full_lightcurves.csv"),
            sep=","
        )

        features = pd.DataFrame({"object_id": split_lc["object_id"].unique()})
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

