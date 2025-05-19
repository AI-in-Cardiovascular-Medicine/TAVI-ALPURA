import os
from os.path import join
import sys
import pandas as pd
import numpy as np
from utils import (remove_patients_without_outcome, remove_0_variance_features, remove_features_with_many_nan,
                   remove_binaries_not_populated)


def clean_dataset(config, to_drop):
    """Cleaning both datasets"""
    df = {}
    features_all_na = []
    features_0var = []
    features_many_nan = []
    features_binary_empty = []

    for dataset in ["Bern", "Japan"]:
        print(f"--- {dataset} ---")
        df[dataset] = pd.read_excel(config["data_path"], sheet_name=dataset)
        df[dataset].columns = df[dataset].columns.str.replace(" ", "").str.replace(".", "_")
        print(f"{df[dataset].shape[0]} patients, {df[dataset].shape[1]} features")

        # Remove patients without time to event or event label
        df[dataset] = remove_patients_without_outcome(df[dataset], config["time_column"], config["outcome"])
        # Replace "Missing" with nan
        df[dataset].replace(to_replace={"no": 0, "yes": 1}, inplace=True)
        df[dataset] = df[dataset].apply(pd.to_numeric, errors='coerce')  # replace non-numeric entries with NaN
        df[dataset] = df[dataset].replace(to_replace="Not Measured", value=np.nan)
        # Set outliers to NaN
        nunique = df[dataset].nunique()
        categorical = list(nunique[nunique < 5].index)
        continuous = [col for col in df[dataset].columns if col not in categorical]
        q1 = df[dataset][continuous].quantile(0.01)
        q3 = df[dataset][continuous].quantile(0.99)
        iqr = q3 - q1
        mask = (df[dataset][continuous] > q3 + iqr) | (df[dataset][continuous] < q1 - iqr)
        for col in continuous:
            if np.sum(mask[col]) > 0:
                print(f"Set {np.sum(mask[col])} nans for feature {col}, values {df[dataset].loc[mask[col], col].values}")
        df[dataset][mask] = np.nan
        # Get columns all empty
        cols_na = df[dataset].columns[df[dataset].isna().all()].values.tolist()
        features_all_na += cols_na
        if len(cols_na) > 0:
            print(f"{len(cols_na)} columns with all NA values in {dataset}:\n\t{cols_na}")
        # Get columns with 0 variance
        _, cols_0var = remove_0_variance_features(df[dataset], return_drop_features=True, verbose=False)
        cols_0var = [col for col in cols_0var if col not in cols_na]
        features_0var += cols_0var
        if len(cols_na) > 0:
            print(f"{len(cols_0var)} columns with 0 variance in {dataset}:\n\t{cols_0var}")
        # Get columns with many NANs
        _, cols_many_nan = remove_features_with_many_nan(df[dataset], nan_threshold=config["nan_threshold"], return_drop_features=True, verbose=False)
        cols_many_nan = [col for col in cols_many_nan if col not in cols_na + cols_0var]
        features_many_nan += cols_many_nan
        if len(cols_many_nan) > 0:
            print(f"{len(cols_many_nan)} columns with more than {100*config['nan_threshold']}% nan in {dataset}:\n\t{cols_many_nan}")
        # Drop binary features not populated (all 0s or 1s)
        _, binary_not_populated = remove_binaries_not_populated(df[dataset], config["binary_threshold"], verbose=False, return_drop_features=True)
        binary_not_populated = [col for col in binary_not_populated if col not in cols_na + cols_0var + cols_many_nan]
        features_binary_empty += binary_not_populated
        if len(binary_not_populated) > 0:
            print(f"{len(binary_not_populated)} binary columns with less than {100*config['binary_threshold']}% 1s in {dataset}:\n\t{binary_not_populated}")

    # Remove features
    features_to_drop = list(set(features_all_na + features_0var + features_many_nan + to_drop + features_binary_empty))
    print(f"\n--> Total of {len(features_to_drop)} columns to drop:\n{features_to_drop}")
    for dataset in ["Bern", "Japan"]:
        df[dataset] = df[dataset].drop(columns=features_to_drop)
    # Aggregate values for Base_DStro
    df["Japan"]["Base_DStro"] = df["Japan"]["Base_DStro"].replace({3: 1})
    # Save
    os.makedirs(config["output_path"], exist_ok=True)
    for dataset in ["Bern", "Japan"]:
        df[dataset].to_excel(os.path.join(config["output_path"], f"{dataset}.xlsx"), index=False)
        print(f"{dataset}: {len(df[dataset])} patients, {df[dataset].shape[1]} features")


def main():
    outcomes = ["OutCome_Death", "OutCome_CDeath"]
    competing_labels = ["event_competing_death", "event_competing_cdeath"]
    time_columns = ["OutCome_Death_Days", "OutCome_Death_Days"]
    data_filenames = ["Mortality_bern_japan.xlsx", "Mortality_bern_japan.xlsx"]
    out_folders = ["death", "cdeath"]
    logs_path = "logs/taviML/recent"
    os.makedirs(logs_path, exist_ok=True)
    logfile = open(join(logs_path, "preprocessing.log"), "w")  # log file
    sys.stdout = logfile  # Redirect standard output to the file
    sys.stderr = logfile
    for outcome, competing_label, time_column, filename, out_folder in zip(outcomes, competing_labels, time_columns, data_filenames, out_folders):
        config = {
            "outcome": outcome,  # OutCome_Death , OutCome_CDeath
            "event_comp_col": competing_label,
            "time_column": time_column,  # OutCome_Death_Days
            "data_path": f"/home/aici/Projects/TAVI_ML/{filename}",  # path to data before pre-processing
            "nan_threshold": .25,
            "binary_threshold": 0.001,
            "output_path": f"datasets/{out_folder}/",
            "output_path_logs": logs_path
        }
        print(f"\n******* {out_folder} *******")
        to_drop = ["Labo_Thr", "ID"]
        outcome_to_drop = ["OutCome_Death", "event_competing_death"] if config["outcome"] == "OutCome_CDeath" else\
            ["OutCome_CDeath", "event_competing_cdeath"]
        to_drop += outcome_to_drop
        os.makedirs(config["output_path"], exist_ok=True)
        clean_dataset(config, to_drop)
    logfile.close()


if __name__ == "__main__":
    main()  # Run the pre-processing for both outcomes
