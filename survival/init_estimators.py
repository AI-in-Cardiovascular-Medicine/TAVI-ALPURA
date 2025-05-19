from .selectors import fit_and_score_features, ManualSelector, SelectKBestCustom, SelectFeaturesByIndices
from .models import CustomCoxnet
from sklearn.feature_selection import SelectKBest, RFE, SequentialFeatureSelector
from sklearn.preprocessing import RobustScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from survival.models import DeepSurv
import numpy as np


def init_scaler(scaler_name):
    scalers_dict = {
        'RobustScaler': RobustScaler(),
    }
    return scalers_dict[scaler_name]


def init_model(seed, n_workers, model_name):
    models_dict = {
        'CoxPH': CoxPHSurvivalAnalysis(n_iter=10000),
        'CoxNet': CustomCoxnet(alpha=0.0001),
        'RSF': RandomSurvivalForest(random_state=seed, n_jobs=n_workers, n_estimators=50, max_depth=5),
        'GBS': GradientBoostingSurvivalAnalysis(random_state=seed, n_estimators=50, max_depth=5),
        'DeepSurv': DeepSurv()
    }
    return models_dict[model_name]


def scorer(estimator, x, y):
    return estimator.score(x, y)


def get_selector_with_feature_indices(selector_init, x, y):
    selector_init.fit(x, y)
    features_indices = selector_init.get_support(indices=True)
    return SelectFeaturesByIndices(feature_indices=features_indices), len(features_indices)


def init_selector(n_workers, selector_name, x, y, model, manually_selected_features=None):
    n_features = None
    try:
        if hasattr(model, 'feature_importances_'):
            estimator = model  # If 'feature_importances_' is implemented, use the model
        else:
            raise AttributeError()  # Raise error if attribute is missing
    except (AttributeError, NotImplementedError) as e:
        estimator = CoxPHSurvivalAnalysis(n_iter=10000)  # Use CoxPH if the model doesn't have 'feature_importances_'
        print("The model does not implement 'feature_importances_', using CoxPH to compute feature importance.")

    # Initialize selector
    if selector_name == "SelectKBest":
        # Pre-compute the list of features ordered by importance
        selector_init = SelectKBest(k="all", score_func=fit_and_score_features)
        selector_init.fit(x, y)
        ordered_indices = np.argsort(selector_init.scores_)[::-1]
        selector = SelectKBestCustom(feature_ordered_indices_=ordered_indices)
    elif selector_name == "RFE":
        selector_init = RFE(estimator, n_features_to_select=int(0.5*x.shape[1]))  # select half of the features
        selector, n_features = get_selector_with_feature_indices(selector_init, x, y)
    elif selector_name == "Manual":
        selector = ManualSelector(features=manually_selected_features)
    elif selector_name == 'SequentialForward':
        selector_init = SequentialFeatureSelector(model, direction="forward", n_jobs=n_workers, n_features_to_select=5)
        selector, n_features = get_selector_with_feature_indices(selector_init, x, y)
    else:
        raise ValueError(f"Selector {selector_name} not valid")
    return selector, n_features
