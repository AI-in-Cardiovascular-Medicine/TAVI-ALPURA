import numpy as np
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint


def set_params_search_space(n_features, search_strategy):
    """
    Defines hyperparameter search space for survival models based on search_strategy.

    Parameters:
    - n_features: int, number of available features
    - search_strategy: str, either "bayes" for BayesSearchCV or "rand" for RandomizedSearchCV

    Returns:
    - model_params: dict, hyperparameter distributions for models
    - selector_params: dict, hyperparameter distributions for feature selectors
    """
    def get_param(value_bayes, value_random): return value_bayes if search_strategy == "bayes" else value_random
    model_params = {
        "CoxNet": {
            "model__alpha": get_param(Real(0.00001, 0.01, prior="log-uniform"), uniform(0.00001, 0.00999)),
            "model__l1_ratio": get_param(Real(0.01, 1), uniform(0.01, 0.99))
        },
        "RSF": {
            "model__n_estimators": get_param(Integer(50, 100), randint(50, 101)),
            "model__max_depth": get_param(Integer(2, 5), randint(2, 6)),
            "model__min_samples_split": get_param(Integer(5, 25), randint(5, 26)),
            "model__min_samples_leaf": get_param(Integer(5, 25), randint(5, 26)),
            "model__max_samples": get_param(Real(0.2, 0.9), uniform(0.2, 0.7)),
            "model__max_features": get_param(Categorical(["log2"]), ["log2"])  # "sqrt"
        },
        "GBS": {
            "model__learning_rate": get_param(Real(0.001, 0.5, prior="log-uniform"), uniform(0.001, 0.499)),
            "model__n_estimators": get_param(Integer(50, 100), randint(50, 101)),
            "model__dropout_rate": get_param(Real(0., 0.9), uniform(0., 0.9)),
            "model__subsample": get_param(Real(0.5, 1), uniform(0.5, 0.5)),
            "model__max_depth": get_param(Integer(2, 5), randint(2, 6)),
            "model__max_features": get_param(Categorical(["log2"]), ["log2"])
        },
        "DeepSurv": {
            "model__num_nodes": get_param(Integer(5, 20), randint(5, 21)),
            "model__epochs": get_param(Integer(5, 100), randint(5, 101)),
            "model__learning_rate": get_param(Real(1e-5, 1e-2, prior="log-uniform"), uniform(1e-5, 0.00999)),
            "model__dropout": get_param(Real(0.1, 0.5), uniform(0.1, 0.4))
        }
    }

    selector_params = {
        "SelectKBest": {
            "selector__k": get_param(Integer(1, np.ceil(0.7*n_features)), randint(1, np.ceil(0.7*n_features) + 1))
        },
        "Manual": {},
        "RFE": {},
        "SequentialForward": {}
    }
    return model_params, selector_params


def set_hyperparams_optimizer(search_strategy, pipeline, param_grid, n_iter_search, stratified_folds, n_workers, seed):
    """
    Create a hyperparameter optimizer using Bayesian or random search.

    Parameters:
    - search_strategy (str): Search type, either "bayes" for BayesSearchCV or "rand" for RandomizedSearchCV.
    - pipeline (Pipeline): scikit-learn pipeline or estimator to optimize.
    - param_grid (dict): Parameter search space.
    - n_iter_search (int): Number of iterations for the search.
    - stratified_folds (int or CV splitter): Cross-validation strategy.
    - n_workers (int): Number of parallel jobs.
    - seed (int): Random seed for reproducibility.

    Returns:
    - optimizer: Configured BayesSearchCV or RandomizedSearchCV object.
    """
    if search_strategy == "bayes":
        optimizer = BayesSearchCV(
            pipeline,
            param_grid,
            n_iter=n_iter_search,
            n_points=n_iter_search,
            return_train_score=True,
            cv=stratified_folds,
            n_jobs=n_workers,
            error_score='raise',
            random_state=seed,
            refit=True
        )
    elif search_strategy == "rand":
        optimizer = RandomizedSearchCV(
            estimator=pipeline,
            n_iter=n_iter_search,
            param_distributions=param_grid,
            n_jobs=n_workers,
            cv=stratified_folds,
            verbose=0)
    else:
        raise ValueError("search_type must be either 'bayes' or 'rand', got '{0}'".format(search_strategy))
    return optimizer
