import numpy as np
from sksurv.functions import StepFunction
from sksurv.base import SurvivalAnalysisMixin
from sksurv.metrics import concordance_index_censored
from scipy.interpolate import interp1d


class SurvivalEnsemble(SurvivalAnalysisMixin):
    """
    Ensemble of pre-fitted survival models that aggregates predictions using weighted averaging.

    Supports survival function, cumulative hazard function, and risk score prediction.
    Designed for compatibility with scikit-survival pipelines.
    """
    def __init__(self, models, weights=None):
        """
        Initialize the SurvivalEnsemble with a list of pre-fitted survival models.

        Parameters:
        - models (list): List of fitted survival models implementing `predict_survival_function`.
        - weights (list or np.ndarray, optional): Weights for each model in the ensemble.
          If None, models are equally weighted.
        """
        self.models = models
        self.weights = weights if weights is not None else np.ones(len(models)) / len(models)
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """
        Dummy fit method to be compatible with scikit-learn's Pipeline.
        Sets feature_names_in_ to ensure compatibility.
        """
        if hasattr(X, "columns"):  # If X is a pandas DataFrame
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            self.feature_names_in_ = np.arange(X.shape[1])  # If X is a NumPy array
        return self  # Return self, as no actual fitting is needed for pre-fitted models

    def predict_survival_function(self, X, return_array=False):
        """
        Predict survival function by averaging the survival probabilities of each model.

        Parameters:
        - X: Features (numpy array or DataFrame)
        - return_array: Boolean flag indicating whether to return the survival function as an array (default is False)

        Returns:
        - If return_array=True: A 2D array of survival probabilities for each time point.
        - If return_array=False: A list of step functions (time points and survival probabilities).
        """
        survival_funcs = [model.predict_survival_function(X) for model in self.models]

        # Use the time points from the first model
        times = survival_funcs[0][0].x  # Assuming the first model has consistent time points

        # Interpolate survival functions from other models to match the time points of the first model
        interpolated_survival = []
        for sf in survival_funcs:
            model_survival = []
            for i in range(len(X)):
                model_times = sf[i].x
                model_survival_values = sf[i].y
                interpolator = interp1d(model_times, model_survival_values, kind='linear', fill_value='extrapolate')
                model_survival.append(interpolator(times))
            interpolated_survival.append(np.array(model_survival))

        # Compute the weighted average of the survival functions
        avg_survival = np.average(interpolated_survival, axis=0, weights=self.weights)

        # Return the result based on the flag
        if return_array:
            return avg_survival  # Return the array of survival probabilities
        else:
            # Return step function: list of StepFunction instances (time points, survival probabilities)
            step_functions = []
            for i in range(len(X)):
                step_func = StepFunction(times, avg_survival[i])
                step_functions.append(step_func)
            return step_functions

    def predict_cumulative_hazard_function(self, X):
        """
        Predict cumulative hazard function by averaging across models.

        Parameters:
        - X: Features (numpy array or DataFrame)

        Returns:
        - List of averaged cumulative hazard functions
        """
        hazard_funcs = [model.predict_cumulative_hazard_function(X) for model in self.models]

        avg_hazard_funcs = []
        for i in range(len(X)):
            times = hazard_funcs[0][i].x  # Use time points from the first model
            avg_hazard = np.average([hf[i].y for hf in hazard_funcs], axis=0, weights=self.weights)
            avg_hazard_funcs.append((times, avg_hazard))

        return avg_hazard_funcs

    def predict(self, X):
        """
        Predict risk scores by averaging model risk scores.

        Parameters:
        - X: Features (numpy array or DataFrame)

        Returns:
        - Array of risk scores
        """
        risks = np.array([model.predict(X) for model in self.models])
        return np.average(risks, axis=0, weights=self.weights)
