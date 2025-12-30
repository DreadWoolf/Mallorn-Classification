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


    data_folder = "Data"
    filename = "MALLORN-data_test.csv"
    data_path = os.path.join(os.getcwd(), data_folder, filename)

    data = pd.read_csv(data_path, sep=',')

    excluded_cols = Stackingmodel.get_excluded_cols
    if excluded_cols:
        excluded_cols = [c for c in excluded_cols if c != "object_id"]
        data = data.drop(columns=excluded_cols, errors="ignore")

    X = data.drop(columns="object_id")
    print(X.isna().sum().sort_values(ascending=False).head(10))

    y_pred = predict(Stackingmodel, X)

    submission_df = pd.DataFrame({
        "object_id": data["object_id"].values,
        "prediction": y_pred.astype(int)
    })

    return submission_df



# def create_predicted_df(model_name):
#     Stackingmodel: StackingEnsemble = StackingEnsemble.load_or_create(model_name)

#     data_folder = "Data"
#     filename = "MALLORN-data_test.csv"
#     data_path = os.path.join(os.path.join(os.getcwd(), data_folder), filename)

    
#     data = pd.read_csv(data_path, sep=',')

#     # data.drop(columns=Stackingmodel.get_excluded_cols != "object_id", inplace=True, errors= "ignore")

#     # Only drop columns the model should not see (but NOT object_id)
#     excluded_cols = Stackingmodel.get_excluded_cols

#     if excluded_cols != None:
#         excluded_cols = [c for c in excluded_cols if c != "object_id"]
#         data = data.drop(columns=excluded_cols, errors="ignore")

#     print(excluded_cols)

#     submisson_df = pd.DataFrame(columns= ("object_id", "prediction"))

#     # print(data.head(4))
#     for i in range(len(data)):
#         vector = data.iloc[[i]]   # already a DataFrame
#         # print(vector, end="\n\n\n")
#         object_id = vector["object_id"].iloc[0]
#         y_pred = predict(Stackingmodel, input_vector=vector.drop(columns="object_id"))
#         # print(y_pred)

#         # submisson_df.merge(vector["object_id"], y_pred)
#         submisson_df.loc[len(submisson_df)] = {
#             # "object_id": vector["object_id"].values[0],
#             "object_id": object_id,
#             "prediction": int(y_pred[0])
#         }
    
#     return submisson_df


    



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