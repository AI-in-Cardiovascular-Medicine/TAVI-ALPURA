import pandas as pd


def remove_features_with_many_nan(data, nan_threshold, return_drop_features=False, verbose=True):
    """
    Removes features with missing values above a threshold.
    """
    null_perc = data.isna().sum() / len(data)
    to_drop = null_perc[null_perc > nan_threshold].index.values.tolist()
    if len(to_drop) > 0 and verbose:
        print(f"Removing {len(to_drop)} features (more than {nan_threshold*100}% of nans):\n\t{to_drop}")
    data = data.drop(columns=to_drop)
    if return_drop_features:
        return data, to_drop
    else:
        return data


def remove_binaries_not_populated(data, binary_threshold=0.01, return_drop_features=False, verbose=True):
    """
    Removes binary features with very low or high prevalence based on a population threshold.
    """
    binary_features = [col for col in data.columns if set(data[col].dropna().unique()) in [{0, 1}, {0}, {1}, {"0", "1"}, {"0"}, {"1"}]]
    data[binary_features] = data[binary_features].apply(pd.to_numeric)
    binary_frac = data[binary_features].sum() / len(data)
    to_drop = binary_frac[(binary_frac < binary_threshold) | (binary_frac > 1-binary_threshold)]
    if len(to_drop) > 0 and verbose:
        print(f"Removing {len(to_drop)} low populated binary features (% < {binary_threshold*100} or % > "
              f"{100-binary_threshold*100}):\n{to_drop*100}")
    data = data.drop(columns=to_drop.index.values)
    if return_drop_features:
        return data, to_drop.index.values.tolist()
    else:
        return data


def remove_0_variance_features(data, return_drop_features=False, verbose=True):
    """
    Remove features with zero variance from the dataset.
    """
    data_var = data.var(numeric_only=True)
    data_var_0 = data_var[data_var == 0].index.values
    if len(data_var_0) > 0 and verbose:
        print(f"Removing {len(data_var_0)} features with 0 variance:\n\t{data_var_0}")
    data = data.drop(columns=data_var_0)
    if return_drop_features:
        return data, data_var_0.tolist()
    else:
        return data


def remove_patients_without_outcome(data, time_column, event_column):
    """
    Remove patients missing time-to-event or event outcome data.
    """
    indices_no_outcome = data[(data[time_column].isna()) | (data[event_column].isna())].index
    if len(indices_no_outcome) > 0:
        print(f"Removing {len(indices_no_outcome)} patients without time to event or event label")
    data = data.drop(indices_no_outcome)
    return data
