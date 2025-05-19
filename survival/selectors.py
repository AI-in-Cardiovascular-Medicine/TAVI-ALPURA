from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
import sksurv.metrics as sksurv_metrics
import numpy as np
import pandas as pd


def fit_and_score_features(X, y, scoring=None):
    """
    Fits a CoxPH model to each individual feature and returns their scores.

    Parameters:
        X (array-like or DataFrame): Feature matrix.
        y (structured array): Survival outcome (event indicator and time).
        scoring (str, optional): Name of a scoring function from sksurv.metrics.

    Returns:
        np.ndarray: Array of scores, one per feature.
    """
    n_features = X.shape[1]
    scores = np.empty(n_features)
    model = CoxPHSurvivalAnalysis(alpha=0.1)
    if scoring is not None:
        estimator = getattr(sksurv_metrics, scoring)(model)  # attach scoring function
    else:
        estimator = model
    for feature in range(n_features):
        if isinstance(X, pd.DataFrame):
            X_feature = X.iloc[:, feature: feature + 1]  # For Pandas DataFrame, select one feature (by column)
        else:
            X_feature = X[:, feature: feature + 1]  # For NumPy array, select one feature (by column index)
        estimator.fit(X_feature, y)
        scores[feature] = estimator.score(X_feature, y)
    return scores


class ManualSelector(BaseEstimator, SelectorMixin, TransformerMixin):
    def __init__(self, features):
        """
        Custom feature selector in scikit-learn style.
        Parameters:
        - features: list of feature names to select from the dataset.
        """
        self.features = features

    def fit(self, X, y=None):
        """
        Fit method (required by scikit-learn but does nothing here).
        Parameters:
        - X: pandas DataFrame or numpy array (input data).
        - y: target values (ignored).
        Returns:
        - self: the fitted object.
        """
        # Ensure the selected features exist in the dataset
        if isinstance(X, pd.DataFrame):
            missing_features = set(self.features) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features in input data: {missing_features}")
        return self

    def transform(self, X):
        """
        Transform method to select features.
        Parameters:
        - X: pandas DataFrame or numpy array (input data).

        Returns:
        - Transformed dataset with selected features.
        """
        if isinstance(X, pd.DataFrame):
            return X[self.features]
        elif isinstance(X, np.ndarray):
            raise TypeError("Feature selection requires pandas DataFrame input.")
        else:
            raise TypeError("Input data must be a pandas DataFrame.")

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit and transform the dataset.
        Parameters:
        - X: pandas DataFrame (input data).
        - y: target values (optional, ignored).

        Returns:
        - Transformed dataset with selected features.
        """
        return self.fit(X, y).transform(X)

    def _get_support_mask(self):
        """
        Required method for SelectorMixin.
        Returns a boolean mask indicating selected features.
        """
        if not hasattr(self, "features"):
            raise ValueError("Features list is not defined.")

        # Create a mask where selected features are True
        return np.array([col in self.features for col in self.feature_names_in_])


class SelectKBestCustom(BaseEstimator, SelectorMixin, TransformerMixin):
    def __init__(self, feature_ordered_indices_, k=10):
        """
        Initialize with a list of ordered feature indices and the number of features to select.
        Parameters:
        - ordered_indices: List or array of feature indices ordered by importance.
        - k: Number of top features to select from the ordered indices.
        """
        self.feature_ordered_indices_ = feature_ordered_indices_  # List of ordered feature indices
        self.k = k  # Number of features to select

    def fit(self, X, y=None):
        """
        Fit method (does nothing in this case).
        """
        # The fit method doesn't need to do anything since ordered_indices are provided during initialization.
        return self

    def transform(self, X):
        """
        Transform the input data by selecting the top k features based on the ordered indices.
        Parameters:
        - X: Input dataset (can be a pandas DataFrame or numpy array).
        Returns:
        - Transformed dataset with the top k features.
        """
        # Ensure that the ordered_indices are valid
        if len(self.feature_ordered_indices_) < self.k:
            raise ValueError("Ordered indices are fewer than the number of features to select (k).")

        # Select the top k features based on the ordered indices
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.feature_ordered_indices_[:self.k]]
        else:
            return X[:, self.feature_ordered_indices_[:self.k]]

    def _get_support_mask(self):
        """
        Return a boolean mask of the selected features.
        This method is required by SelectorMixin to return the selected features as a mask.
        """
        # Create a mask of all False (not selected)
        mask = [False] * len(self.feature_ordered_indices_)

        # Set True for the top k selected features
        for idx in self.feature_ordered_indices_[:self.k]:
            mask[idx] = True


class SelectFeaturesByIndices(SelectorMixin, BaseEstimator):
    """
    Feature selector that selects features based on specified column indices.

    Parameters:
        feature_indices (list or array-like): Indices of features to select.
    """
    def __init__(self, feature_indices):
        self.feature_indices = feature_indices

    def fit(self, X, y=None):
        if max(self.feature_indices) >= X.shape[1]:
            raise ValueError("One or more feature indices are out of range.")
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.feature_indices]
        elif isinstance(X, np.ndarray):
            return X[:, self.feature_indices]
        else:
            raise TypeError("Input X must be a pandas DataFrame or a numpy array.")

    def _get_support_mask(self):
        """
        Returns a boolean mask where selected features are True.
        """
        mask = np.zeros(self.feature_names_in_.shape[0], dtype=bool)
        mask[self.feature_indices] = True
        return mask
