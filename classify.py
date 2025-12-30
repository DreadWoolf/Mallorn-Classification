import pandas as pd
import numpy as np
from StackingEnsemble import StackingEnsemble
import os





main_folder = "mallorn-astronomical-classification-challenge"
data_folder = "Data"
data_path = os.path.join(os.getcwd(), data_folder)

# pathos = "year3/Machine_Learning/Final_project"

# path = os.path.join(os.getcwd(), pathos)
# path = os.path.join(path, main_folder)

path = os.path.join(os.getcwd(), main_folder)

def create_submissionfile(model_name = "saved_model"):
    # Create the dataframe with predictions
    submission_df = create_predicted_df(model_name)

    # Define submission folder
    submission_folder = os.path.join(os.getcwd(), "submission_files")

    # Create folder if it does not exist
    os.makedirs(submission_folder, exist_ok=True)

    # Count existing submission files
    existing_files = [
        f for f in os.listdir(submission_folder)
        if f.startswith("submission_") and f.endswith(".csv")
    ]

    # Determine next file number
    submission_number = len(existing_files)

    # Build filename
    filename = f"submission_{submission_number}.csv"
    file_path = os.path.join(submission_folder, filename)

    # Save CSV
    submission_df.to_csv(file_path, index=False)

    print(f"Submission saved to: {file_path}")

    return submission_df




def create_predicted_df(model_name):
    Stackingmodel = StackingEnsemble.load_or_create(model_name)

    data_path = os.path.join(os.getcwd(), "Data", "MALLORN-data_test.csv")
    data = pd.read_csv(data_path)

    excluded_cols = Stackingmodel.get_excluded_cols
    if excluded_cols:
        excluded_cols = [c for c in excluded_cols if c != "object_id"]
        data = data.drop(columns=excluded_cols, errors="ignore")


    valid_mask = ~data.isna().any(axis=1)
    data_safe = data.loc[valid_mask]
    data_nan  = data.loc[~valid_mask]

    X_safe = data_safe.drop(columns="object_id")
    X_nan  = data_nan.drop(columns="object_id")

    print(f"Safe rows: {len(X_safe)}, NaN rows: {len(X_nan)}")

    df_safe_pred = pd.DataFrame({
        "object_id": data_safe["object_id"],
        "prediction": predict(Stackingmodel, X_safe).astype(int)
    }, index=data_safe.index)

    df_nan_pred = pd.DataFrame({
        "object_id": data_nan["object_id"],
        "prediction": predict(Stackingmodel, X_nan).astype(int)
    }, index=data_nan.index)

    submission_df = (
        pd.concat([df_safe_pred, df_nan_pred])
        .sort_index()
        .reset_index(drop=True)
    )

    return submission_df
    



def predict(Stackingmodel: StackingEnsemble, input_vector):

    if "Z_err" not in input_vector.columns:
        return Stackingmodel.predict(input_vector)

    z_error = input_vector["Z_err"]
    v_no_error = input_vector.drop(columns=["Z_err"]).copy()

    v_z_err_sub = v_no_error.copy()
    v_z_err_add = v_no_error.copy()

    # v_z_err_sub["z"] = max(0, input_vector["z"] - z_error)
    # v_z_err_add["z"] = input_vector["z"] + z_error

    v_z_err_sub["Z"] = np.maximum(0.0, v_no_error["Z"] - z_error)
    v_z_err_add["Z"] = v_no_error["Z"] + z_error


    y_sub = Stackingmodel.predict(v_z_err_sub)
    y_nom = Stackingmodel.predict(v_no_error)
    y_add = Stackingmodel.predict(v_z_err_add)

    votes = y_sub + y_nom + y_add
    y_final = (votes >= 2).astype(int)

    return y_final






if __name__ == "__main__":

    print(os.getcwd())

    create_submissionfile()