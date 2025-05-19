import os
import sys
import pickle
import time

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from survival.ensemble import SurvivalEnsemble
from survival.init_estimators import init_scaler, init_model, init_selector
from survival.hyperparameters_search import set_params_search_space, set_hyperparams_optimizer
from helpers.nested_dict import NestedDefaultDict
from evaluation.calibration import mean_calibration, ici_survival_times
from evaluation.discrimination import antolini_concordance_index
from sklearn.utils import resample  # for Bootstrap sampling
from joblib import Parallel, delayed


class Survival:
    """
    Trains and evaluates multiple machine learning survival models.
    Supports different scalers, feature selectors, models, and hyperparameter search strategies.
    Each model is trained via cross-validation, and their predictions are combined into an ensemble.
    The ensemble is then evaluated on an external test set using multiple survival metrics.
    """
    def __init__(self, config, progress_manager=None) -> None:
        self.progress_manager = progress_manager
        self.out_dir = config.meta.out_dir
        self.table_file = os.path.join(self.out_dir, 'results_table.xlsx')
        self.results_file = os.path.join(self.out_dir, 'results.pkl')
        self.event_column = config.meta.events
        self.time_column = config.meta.times
        self.n_workers = config.meta.n_workers
        self.scalers = [scaler for scaler in config.survival.scalers if config.survival.scalers[scaler]]
        self.selectors = [sel for sel in config.survival.feature_selectors if config.survival.feature_selectors[sel]]
        self.models = [model for model in config.survival.models if config.survival.models[model]]
        self.n_cv_splits = config.survival.n_cv_splits
        self.n_iter_search_bayes = config.survival.n_iter_search_bayes
        self.n_iter_search_rand = config.survival.n_iter_search_rand
        self.eval_times = np.array(config.evaluation.eval_times)
        self.eval_times_names = config.evaluation.eval_times_names
        self.tau = self.eval_times[-1]  # truncation time
        self.bootstrap_iterations = config.evaluation.bootstrap_iterations

        search_strategies_dict = config.survival.hyperparams_search_strategy
        self.search_strategies = [strategy for strategy in search_strategies_dict if search_strategies_dict[strategy]]

        if "Manual" in self.selectors:
            if config.meta.manually_selected_features is None:
                logger.info(f"''Manual'' feature selection needs config.meta.manually_selected_features to be"
                            f"specified. Manual feature selection will not be performed.")
                self.selectors.remove("Manual")
            else:
                self.manually_selected_features = config.meta.manually_selected_features
        else:
            self.manually_selected_features = None

        self.total_combinations = (
                len(self.scalers)
                * len(self.selectors)
                * len(self.models)
        )
        set_names = ["val", "test"]
        self.result_cols = (
                ["Scaler", "Selector", "Model", 'search', 'train_strategy',
                 "harrell_val", "CI_harrell_val", "uno_val", "CI_uno_val", "ant_val", "CI_ant_val",
                 "harrell_test", "CI_harrell_test", "uno_test", "CI_uno_test", "ant_test", "CI_ant_test"] +
                [f"auc_{time_name}_{split}" for split in set_names for time_name in self.eval_times_names] +
                [f"CI_auc_{time_name}_{split}" for split in set_names for time_name in self.eval_times_names] +
                [f"mCalib_{time_name}_{split}" for split in set_names for time_name in self.eval_times_names] +
                [f"CI_mCalib_{time_name}_{split}" for split in set_names for time_name in self.eval_times_names] +
                [f"ici_{time_name}_{split}" for split in set_names for time_name in self.eval_times_names] +
                [f"CI_ici_{time_name}_{split}" for split in set_names for time_name in self.eval_times_names] +
                ["brier_score_test", 'evaluation_times', 'truncation_time', 'n_iter_search', "Seed"]
        )
        self.results_table = pd.DataFrame(index=range(self.total_combinations), columns=self.result_cols)
        self.row_to_write = 0
        self.results = NestedDefaultDict()

    def __call__(self, seed, x_train, y_train, x_test, y_test, comp_label_train, comp_label_test):
        # Initialize parameters
        self.seed = seed
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.results[self.seed]['x_train'] = x_train
        self.results[self.seed]['y_train'] = y_train
        self.results[self.seed]['x_test'] = x_test
        self.results[self.seed]['y_test'] = y_test
        self.results[self.seed]['comp_label_train'] = comp_label_train
        self.results[self.seed]['comp_label_test'] = comp_label_test
        self.results[self.seed]['eval_times'] = self.eval_times
        self.results[self.seed]['eval_times_names'] = self.eval_times_names
        # Update manually selected features based on preprocessing
        if self.manually_selected_features is not None:
            self.manually_selected_features = np.intersect1d(self.manually_selected_features,
                                                             x_train.columns.values).tolist()
        # Fit and evaluate pipeline
        self.fit_and_evaluate_pipelines()
        return self.results_table

    def fit_selector(self, selector_name, scaler, estimator):
        """
        Fits the feature selector and returns the fitted selector and the number of selected features.
        If an error occurs during fitting, it logs the error and returns None, None.
        """
        selectors_to_fit = ["SelectKBest", "RFE", "SequentialForward"]
        if selector_name in selectors_to_fit:
            logger.info(f"                  Fitting {selector_name} feature selector")
        try:
            tic = time.time()
            selector, n_selected_features = init_selector(
                self.n_workers, selector_name,
                scaler.fit_transform(self.x_train),
                self.y_train, estimator,
                self.manually_selected_features
            )
            selector.set_output(transform="pandas")
            if selector_name in selectors_to_fit:
                logger.info(f"                  Fitting-time {time.time() - tic}")
            if n_selected_features is not None:
                logger.info(f"                  {n_selected_features} features selected")
            return selector
        except Exception as e:
            logger.error(f"                 Error during scaler fitting, scaler is skipped. Error message: {e}")
            return "skip"

    def fit_and_evaluate_pipelines(self):
        pbar = self.progress_manager.counter(
            total=self.total_combinations, desc="Training and evaluating all combinations", unit='it', leave=False
        )
        # Create CV splits
        cv = StratifiedKFold(n_splits=self.n_cv_splits, random_state=self.seed, shuffle=True)
        stratified_folds = [x for x in cv.split(self.x_train, self.y_train[self.event_column])]
        # Loop over models
        for model_name in self.models:
            estimator = init_model(self.seed, self.n_workers, model_name)
            logger.info(f"\033[95m{model_name} model\033[0m")
            for scaler_name in self.scalers:  # Loop over scalers
                scaler = init_scaler(scaler_name)
                logger.info(f"  \033[93m{scaler_name} scaler\033[0m")
                scaler = scaler.set_output(transform="pandas")
                for selector_name in self.selectors:  # Loop over Selectors
                    logger.info(f"    \033[92m{selector_name} selector\033[0m")
                    selector = self.fit_selector(selector_name, scaler, estimator)
                    if selector == "skip":
                        continue
                    for search in self.search_strategies:  # Loop over search strategies
                        logger.info(f"      \033[94m{search} hyperparameters search strategy\033[0m")
                        n_iter_search = {
                            "bayes": self.n_iter_search_bayes,
                            "rand": self.n_iter_search_rand
                        }.get(search, None)
                        try:
                            logger.info(f"      Training {scaler_name} - {selector_name} - {model_name} - {search}")
                            tic = time.time()
                            # Create pipeline and parameter grid
                            pipe = Pipeline(
                                [
                                    ('scaler', scaler),
                                    ("selector", selector),
                                    ("model", estimator),
                                ]
                            )
                            if search == "none":
                                validation_scores = cross_validate(pipe, self.x_train, self.y_train, n_jobs=1,
                                                                   scoring=self.custom_survival_scorer,
                                                                   cv=stratified_folds, return_estimator=True)
                                pipe.fit(self.x_train, self.y_train)
                                refitted_estimator = pipe
                            else:
                                # Hyperparameters search and fitting
                                self.model_params, self.selector_params = set_params_search_space(
                                    n_features=self.x_train.shape[1],
                                    search_strategy=search
                                )
                                param_grid = {**self.selector_params[selector_name], **self.model_params[model_name]}
                                gcv = set_hyperparams_optimizer(
                                    search_strategy=search,
                                    pipeline=pipe,
                                    param_grid=param_grid,
                                    n_iter_search=n_iter_search,
                                    stratified_folds=stratified_folds,
                                    n_workers=self.n_workers,
                                    seed=self.seed
                                )
                                gcv.fit(self.x_train, self.y_train)
                                best_params = gcv.best_params_
                                pipe.set_params(**best_params)
                                # Get scores on validation sets and save
                                validation_scores = cross_validate(pipe, self.x_train, self.y_train,
                                                                   cv=stratified_folds,
                                                                   scoring=self.custom_survival_scorer, n_jobs=1,
                                                                   return_estimator=True)
                                refitted_estimator = gcv.best_estimator_  # extract best estimator
                            # Save validation risk
                            risk_all = self.get_validation_risk(models=validation_scores["estimator"],
                                                                cv_splits=stratified_folds)
                            self.results[self.seed]["val_risk"][scaler_name][selector_name][model_name][
                                search] = risk_all
                            # Create ensemble model
                            models = [validation_scores["estimator"][i]["model"] for i in
                                      range(len(validation_scores["estimator"]))]
                            ensemble_estimator = Pipeline([
                                ('scaler', scaler),
                                ("selector", selector),
                                ("model", SurvivalEnsemble(models=models))
                            ])
                            ensemble_estimator.fit(self.x_train, self.y_train)  # single models are NOT fitted
                            # Evaluate ensemble and refitted models
                            logger.info(f'      Evaluating {scaler_name} - {selector_name} - {model_name} - {search}')
                            row = {"Seed": self.seed, "Scaler": scaler_name, "Selector": selector_name,
                                   "Model": model_name}
                            for strategy, estimator_ in zip(["refit", "ensemble"],
                                                            [refitted_estimator, ensemble_estimator]):
                                row["train_strategy"] = strategy
                                self.results[self.seed][scaler_name][selector_name][model_name][search][
                                    strategy] = estimator_
                                metrics = self.evaluate_model(estimator=estimator_)
                                row.update(metrics)
                                row.update({"n_iter_search": n_iter_search, "search": search})
                                self.results_table.loc[self.row_to_write] = row
                                for column in self.results_table.columns:
                                    self.results_table[column] = pd.to_numeric(self.results_table[column],
                                                                               errors='ignore')
                                self.results_table[self.results_table.select_dtypes(
                                    include='number').columns] = self.results_table.select_dtypes(
                                    include='number').round(3)
                                self.row_to_write += 1
                            self.results_table = self.results_table.sort_values(["Seed", "Scaler", "Selector", "Model",
                                                                                 "search", "train_strategy"])
                            logger.info(f'      Saving results to {self.out_dir}')
                            try:  # ensure that intermediate results are not corrupted by KeyboardInterrupt
                                self.save_results()
                            except KeyboardInterrupt:
                                logger.warning('Keyboard interrupt detected, saving results before exiting...')
                                self.save_results()
                                sys.exit(130)
                            pbar.update()
                            logger.info(f"      Elapsed time {time.time() - tic}")
                        except Exception as e:
                            logger.error(
                                f"      Error encountered for {scaler_name} - {selector_name} - {model_name} - "
                                f"{search}. Error message: {e}")
        pbar.close()

    def get_validation_risk(self, models, cv_splits):
        """
        Computes risk predictions on validation folds from cross-validated models.
        Applies preprocessing, extracts risk at evaluation times, and aggregates results across all folds.

        Parameters:
        - models: List of dicts with 'scaler', 'selector', and fitted 'model'.
        - cv_splits: List of (train_idx, val_idx) tuples.

        Returns:
        - DataFrame with risks, survival targets, and event labels.
        """
        x_train_preprocessed = models[0]["scaler"].transform(self.x_train)
        x_train_preprocessed = models[0]["selector"].transform(x_train_preprocessed)
        comp_labels = self.results[self.seed]['comp_label_train']
        risk_times_all = []
        risk_all = []
        y_all = []
        y_comp_all = []
        for i, model in enumerate(models):
            model = model["model"]  # extract only model without pre-processing steps
            x_val = x_train_preprocessed.iloc[cv_splits[i][1]]
            y_val = self.y_train[cv_splits[i][1]]
            risk_times_all.append(self.get_risk_at_eval_times(model, x_val))
            risk_all.append(model.predict(x_val))
            y_all.append(y_val)
            y_comp_all.append(comp_labels.iloc[cv_splits[i][1]].values)
        self.risk_time_val = np.concatenate(risk_times_all)
        self.risk_val = np.concatenate(risk_all)
        self.y_val = np.concatenate(y_all)
        risk_times_all = pd.DataFrame(self.risk_time_val, columns=self.eval_times_names)
        y_all = pd.concat([pd.DataFrame(a) for a in y_all], ignore_index=True)
        risk_times_all = pd.concat([risk_times_all, y_all], axis=1)
        risk_times_all["risk"] = self.risk_val
        risk_times_all["comp_event_label"] = np.concatenate(y_comp_all)
        return risk_times_all

    def evaluate_model(self, estimator):
        """
        Evaluates a survival model on validation and test sets using multiple metrics. Also bootstrapped confidence
        intervals are computed.

        Parameters:
        - estimator: Trained survival model.

        Returns:
        - Dictionary of evaluation metrics for validation and test sets.
        """
        risk_test = estimator.predict(self.x_test)  # Compute risk scores on test
        risk_time_test = self.get_risk_at_eval_times(estimator, self.x_test)
        # Compute metrics on validation and test
        val_metrics = self.custom_survival_scorer(estimator, X=None, y=self.y_val, risk_scores=self.risk_val,
                                                  risk_at_times=self.risk_time_val, suffix="val")
        test_metrics = self.custom_survival_scorer(estimator, X=None, y=self.y_test, risk_scores=risk_test,
                                                   risk_at_times=risk_time_test, suffix="test")
        metrics_dict = {
            'evaluation_times': self.eval_times.tolist(),
            'truncation_time': self.tau,
            'bootstrap_iterations': self.bootstrap_iterations,
        }
        # Merge all metrics
        metrics_dict.update(val_metrics)
        metrics_dict.update(test_metrics)
        metrics_dict.update(self.bootstrap(risk_test, risk_time_test, self.y_test, suffix="test"))  # CI test set
        metrics_dict.update(self.bootstrap(self.risk_val, self.risk_time_val, self.y_val, suffix="val"))  # CI val set
        return metrics_dict

    def get_risk_at_eval_times(self, estimator, x):
        """
        Computes risk scores at predefined evaluation times.

        Parameters:
        - estimator: Trained survival model.
        - x: Input features.

        Returns:
        - Array of risk scores (1 - survival probability) at each evaluation time.
        """
        surv_func = estimator.predict_survival_function(x)
        risk_scores = np.array([[1 - func(t) for t in self.eval_times] for func in surv_func])  # risk = 1 - surv
        return risk_scores

    @staticmethod
    def bootstrap_step(risk, risk_time, y, y_train, eval_times, tau, time_column, event_column):
        """
        Computes bootstrap estimates of survival metrics on resampled data. Static method to allow parallel processing.

        Parameters:
        - risk: Array of risk scores.
        - risk_time: Array of risk scores at evaluation times.
        - y: Test set survival data (DataFrame).
        - y_train: Training set survival data (for IPCW-based metrics).
        - eval_times: Array of evaluation time points.
        - tau: Truncation time for Uno's c-index.
        - time_column: Column name for event times.
        - event_column: Column name for event indicators.

        Returns:
        - Tuple of computed metrics: Harrell's c-index, Uno's c-index, Antolini's c-index, time-dependent AUC,
          mean calibration, and ICI.
        """
        # Sampling with replacement
        idx_resample = resample(np.arange(len(y)))
        y_resampled = y[idx_resample]
        risk_resampled = risk[idx_resample]
        risk_time_resampled = risk_time[idx_resample]
        # Compute metrics
        cindex_harrell = concordance_index_censored(y_resampled[event_column], y_resampled[time_column],
                                                    risk_resampled)[0]
        cindex_uno = concordance_index_ipcw(y_train, y_resampled, risk_resampled, tau=tau)[0]
        cindex_antolini = antolini_concordance_index(durations=y_resampled[time_column],
                                                     labels=y_resampled[event_column], cuts=eval_times,
                                                     risk=risk_time_resampled, time_max=np.max(eval_times))
        roc_auc = cumulative_dynamic_auc(y_train, y_resampled, risk_time_resampled, eval_times)[0]
        mean_calib = mean_calibration(durations=y_resampled[time_column], eval_times=eval_times,
                                      events=y_resampled[event_column], risk_scores=risk_time_resampled)
        ici = ici_survival_times(durations=y_resampled[time_column], events=y_resampled[event_column],
                                 risk_times=risk_time_resampled, times=eval_times, parallel=False)
        # brier_score = integrated_brier_score(y_train, y_resampled, 1 - risk_time_resampled, eval_times)
        return cindex_harrell, cindex_uno, cindex_antolini, roc_auc, mean_calib, ici  # , brier_score

    def bootstrap(self, risk, risk_time, y, suffix: str = None):
        """
        Computes bootstrap confidence intervals for survival metrics.

        Parameters:
        - risk: Array of risk scores.
        - risk_time: Risk scores at evaluation times.
        - y: Survival labels for the evaluation set.
        - suffix: Optional suffix to append to result keys (e.g. "val" or "test").

        Returns:
        - Dictionary of median and 95% confidence intervals for c-indexes, AUC,
          mean calibration, and ICI.
        """
        args = (risk, risk_time, y, self.y_train, self.eval_times, self.tau, self.time_column, self.event_column)
        boot_results = Parallel(n_jobs=-1)(
            delayed(self.bootstrap_step)(*args) for _ in range(self.bootstrap_iterations))
        # Define metric names and their corresponding indices in boot_results
        metrics = {
            'harrell': 0,
            'uno': 1,
            'ant': 2,
            'auc': 3,
            'mCalib': 4,
            'ici': 5,
        }
        # Computing mean and CI
        ci_dict = {}
        for metric, idx in metrics.items():
            boot_values = np.array([result[idx] for result in boot_results])
            ci_value = [
                np.quantile(boot_values, 0.025, axis=0),
                np.quantile(boot_values, 0.975, axis=0)
            ]
            median = np.quantile(boot_values, 0.5, axis=0)
            setattr(self, f"{metric}_mean", np.mean(boot_values, axis=0))
            setattr(self, f"{metric}_ci", ci_value)
            # Save to metrics dictionary
            if metric in ['auc', 'mCalib', 'ici']:
                for i, time_name in enumerate(self.eval_times_names):
                    ci_dict[f"CI_{metric}_{time_name}"] = [float(np.round(ci_value[0][i], 3)),
                                                           float(np.round(ci_value[1][i], 3))]
                    ci_dict[f"{metric}_{time_name}_median_boot"] = np.round(median[i], 3)
            else:
                ci_dict[f"CI_{metric}"] = [float(np.round(ci_value[0], decimals=3)),
                                           float(np.round(ci_value[1], decimals=3))]
                ci_dict[f"{metric}_median_boot"] = np.round(median, 3)
        if suffix:  # Add suffix to keys if required
            ci_dict = {f"{k}_{suffix}": v for k, v in ci_dict.items()}
        return ci_dict

    def custom_survival_scorer(self, estimator, X, y, risk_scores=None, risk_at_times=None, suffix: str = None):
        """
        Custom scoring function for survival models, computing:
        - Harrell's, UNo's and Antolini's C-index
        - Cumulative Dynamic AUC
        - Mean calibration
        - Integrated Calibration Index (ICI)

        Args:
            estimator: A fitted survival model following scikit-survival API.
            X: Feature matrix.
            y: Survival structured array or DataFrame (event indicator and time).
            risk_scores: Optional, pre-computed risk scores.
            risk_at_times: Optional, pre-computed time-dependent risk scores.
            suffix: Optional suffix to append to metric names.

        Returns:
            A dictionary containing survival evaluation metrics.
        """
        if risk_scores is None:
            risk_scores = estimator.predict(X)  # Compute risk scores
        cindex_harrell = concordance_index_censored(y[self.event_column], y[self.time_column], risk_scores)[
            0]  # Harrell's C-index
        cindex_uno = concordance_index_ipcw(self.y_train, y, risk_scores, tau=self.tau)[
            0]  # Uno's C-index (IPCW-adjusted)
        # Compute time-dependent metrics
        if risk_at_times is None:
            surv_funcs = estimator.predict_survival_function(X)
            risk_at_times = np.array([[1 - sf(t) for t in self.eval_times] for sf in surv_funcs])
        auc_values, _ = cumulative_dynamic_auc(self.y_train, y, risk_at_times, self.eval_times)
        cindex_antolini = antolini_concordance_index(durations=y[self.time_column], labels=y[self.event_column],
                                                     cuts=self.eval_times, risk=risk_at_times,
                                                     time_max=np.max(self.eval_times))
        mean_calibration_values = mean_calibration(durations=y[self.time_column], events=y[self.event_column],
                                                   risk_scores=risk_at_times, eval_times=self.eval_times)
        ici_values = ici_survival_times(durations=y[self.time_column], events=y[self.event_column],
                                        risk_times=risk_at_times, times=self.eval_times)
        brier_score = integrated_brier_score(self.y_train, y, 1 - risk_at_times, self.eval_times)
        metrics_dict = {  # Store results
            "harrell": cindex_harrell,
            "uno": cindex_uno,
            "ant": cindex_antolini,
            "brier_score": brier_score
        }
        for i, time_name in enumerate(self.eval_times_names):
            metrics_dict.update({
                f"auc_{time_name}": auc_values[i],
                f"mCalib_{time_name}": mean_calibration_values.values[i],
                f"ici_{time_name}": ici_values[i]
            })
        if suffix:  # Add suffix to keys if required
            metrics_dict = {f"{k}_{suffix}": v for k, v in metrics_dict.items()}
        return metrics_dict

    def save_results(self):
        """
        Saves the results table as an Excel file and the full results dictionary as a pickle file.
        """
        os.makedirs(os.path.dirname(self.table_file), exist_ok=True)
        self.results_table.to_excel(self.table_file, index=False)
        with open(self.results_file, 'wb') as file:
            pickle.dump(self.results, file)
