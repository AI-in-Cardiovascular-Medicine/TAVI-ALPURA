from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from pycox.models import CoxPH
from sklearn.base import BaseEstimator
from sksurv.base import SurvivalAnalysisMixin
import torchtuples as tt
from torchtuples.practical import MLPVanilla
import numpy as np
from survival.utils import _array_to_step_function
import pandas as pd


class CustomCoxnet(CoxnetSurvivalAnalysis):
    """Wrap class of CoxnetSurvivalAnalysis to enable the optimization of a single alpha"""
    def __init__(self, alpha=0.1, l1_ratio=0.5, **kwargs):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        super(CustomCoxnet, self).__init__(alphas=[alpha], l1_ratio=l1_ratio, fit_baseline_model=True, **kwargs)


class BaseDeepEstimator(SurvivalAnalysisMixin, BaseEstimator):
    """
    Base class for deep survival models with concordance index scoring.
    """
    def __init__(self):
        super().__init__()
        self.model = None

    def predict(self, x):
        raise NotImplementedError

    def score(self, X, y):
        risk_scores = self.predict(X)
        name_event, name_time = y.dtype.names
        return concordance_index_censored(y[name_event], y[name_time], risk_scores)[0]

    def score_split(self, x, duration, events):
        return self.score(x, np.hstack([duration.reshape((-1, 1)), events.reshape((-1, 1))]))


class DeepSurv(BaseDeepEstimator):
    """
    DeepSurv implementation using a multilayer perceptron.
    """
    def __init__(
            self,
            learning_rate=1e-4,
            batch_norm=True,
            dropout=0.0,
            num_nodes=16,
            batch_size=128,
            epochs=10,
            output_features=10,
            early_stopping_patience=10,
            device="cpu"
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.num_nodes_layers = (num_nodes, 2*num_nodes, num_nodes)
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_features = output_features
        self._net = self.model = None
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.feature_names_in_ = None
        self.is_fitted_ = False

    def fit(self, X, y, verbose=False):
        self._net = MLPVanilla(
            in_features=X.shape[1],
            num_nodes=self.num_nodes_layers,
            out_features=1,
            batch_norm=self.batch_norm,
            dropout=self.dropout)
        self.model = CoxPH(net=self._net, optimizer=tt.optim.Adam(self.learning_rate), device=self.device)
        self.model.set_device(self.device)
        y_train = (np.array([item[1] for item in y]), np.array([item[0] for item in y]).astype(np.int64))
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
            X = X.values
        else:
            self.feature_names_in_ = np.arange(X.shape[1])  # If x is a NumPy array
        self.model.fit(
            X.astype("float32"),
            y_train,
            self.batch_size,
            self.epochs,
            verbose=verbose,
        )
        self.is_fitted_ = True
        self.model.compute_baseline_hazards()
        return self

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        risk_scores = np.exp(self.model.predict(x.astype("float32")))
        return risk_scores[:, 0]

    def predict_survival_function(self, X, return_array=False):
        if isinstance(X, pd.DataFrame):
            X = X.values
        surv_df = self.model.predict_surv_df(X.astype("float32"))
        surv = surv_df.transpose().values
        time_points = surv_df.index.values
        if return_array:
            return surv
        return _array_to_step_function(time_points, surv)
