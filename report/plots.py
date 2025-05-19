import os
import warnings
import json
import pickle
from loguru import logger
from ast import literal_eval

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from lifelines import KaplanMeierFitter, AalenJohansenFitter
from lifelines.plotting import add_at_risk_counts
from evaluation.calibration import calibration_plot_survival


plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})


class MakePlots:
    def __init__(self, config) -> None:
        self.plot_format = config.meta.plot_format
        self.time_column = config.meta.times
        self.event_column = config.meta.events
        self.experiment_dir = config.meta.out_dir
        self.dpi = config.meta.plot_dpi
        self.results_file = os.path.join(self.experiment_dir, 'results.pkl')
        self.metrics_table = pd.read_excel(os.path.join(self.experiment_dir, 'results_table_scores.xlsx'))
        self.seed = config.meta.seed
        with open(os.path.join(self.experiment_dir, "best_combinations.json"), "r") as file:
            self.best_combinations = json.load(file)
        with open(os.path.join(self.experiment_dir, "best_combinations_scores.json"), "r") as file:
            self.best_combinations_scores = json.load(file)
        self.search_strategies = [strategy for strategy in config.survival.hyperparams_search_strategy
                                  if config.survival.hyperparams_search_strategy[strategy]]
        self.colors = cm.tab10.colors

    def __call__(self):
        self.out_dir = os.path.join(self.experiment_dir, "plots")
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.results_file, 'rb') as f:
            self.results = pickle.load(f)
        seed = self.seed
        self.x_test = self.results[seed]['x_test']
        self.y_test = self.results[seed]['y_test']
        self.eval_times = self.results[seed]['eval_times']
        self.eval_times_names = self.results[seed]['eval_times_names']
        self.competing_events_label_test = self.results[seed]["comp_label_test"]
        # Plot CIF stratified by risk and calibration curves for all models (no scores)
        combinations = [tuple(comb.values()) for comb in self.best_combinations]
        self.calib_curves = []
        self.labels = []
        for comb in combinations:
            scaler, selector, model, search, train_strategy = comb
            if self.results[seed][scaler][selector][model][search][train_strategy] == {}:
                logger.info(f"No trained model for Scaler={scaler}, Selector={selector},"
                            f"Model={model}, Search={search}.")
                continue
            self.best_estimator = self.results[seed][scaler][selector][model][search][train_strategy]
            # Get Risk scores
            self.risk_scores_test = self.best_estimator.predict(self.x_test)
            df_val_risk = self.results[seed]["val_risk"][scaler][selector][model][search]
            self.risk_scores_val = self.results[seed]["val_risk"][scaler][selector][model][search]["risk"]
            self.competing_events_label_val = self.results[seed]["val_risk"][scaler][selector][model][search]["comp_event_label"]
            self.y_val = self.to_structured_array(df_val_risk[[self.event_column, self.time_column]])
            # CIF
            self.km_cif_by_risk(scaler, selector, model, search, curve_type="cif")
            # Calibration plots
            surv_func = self.best_estimator.predict_survival_function(self.x_test)
            self.risk_scores_times_test = np.array(
                [[1 - func(t) for t in self.eval_times] for func in surv_func])  # risk = 1 - surv
            self.risk_scores_times_val = df_val_risk.drop(columns=[self.event_column, self.time_column, "risk", "comp_event_label"]).values
            calib_curve = self.calibration_plot(scaler, selector, model, search, train_strategy)
            self.calib_curves.append(calib_curve)
            self.labels.append(model)
        self.calibration_plot_all()  # Print calibration for all models

        # AUC for all models and scores
        self.labels = []
        self.auc_all = []
        self.auc_err_all = []
        combinations = [tuple(comb.values()) for comb in self.best_combinations_scores]
        for comb in combinations:
            scaler, selector, model, search, train_strategy = comb
            self.best_estimator = self.results[seed][scaler][selector][model][search][train_strategy]
            auc, auc_err = self.get_cumulative_dynamic_auc(scaler, selector, model, search, train_strategy)
            self.auc_all.append(auc)
            self.auc_err_all.append(auc_err)
            label = model if selector not in ["STS-PROM", "Log ES", "ES 2"] else selector
            self.labels.append(label)
        self.plot_cumulative_dynamic_auc_all()

    def get_cumulative_dynamic_auc(self, scaler, selector, model, search, train_strategy):
        auc_dict = {}
        auc_err_dict = {}
        for set_ in ["val", "test"]:
            mask = (self.metrics_table[["Scaler", "Selector", "Model", "search", "train_strategy"]]
                    .eq([scaler, selector, model, search, train_strategy]).all(axis=1))
            row = self.metrics_table[mask]
            auc = [row[f"auc_{time_name}_{set_}"].values[0] for time_name in self.eval_times_names]
            auc_low = [literal_eval(row[f"CI_auc_{time_name}_{set_}"].values[0])[0] for time_name in self.eval_times_names]
            auc_up = [literal_eval(row[f"CI_auc_{time_name}_{set_}"].values[0])[1] for time_name in self.eval_times_names]
            # Calculate error bars
            auc_err_lower = [auc_val - low for auc_val, low in zip(auc, auc_low)]
            auc_err_upper = [up - auc_val for auc_val, up in zip(auc, auc_up)]
            auc_err = np.array([auc_err_lower, auc_err_upper])
            # Save auc and auc_err
            auc_dict[set_] = np.array(auc)
            auc_err_dict[set_] = np.array(auc_err)
        return auc_dict, auc_err_dict

    def plot_cumulative_dynamic_auc_all(self):
        for set_ in ["val", "test"]:
            plt.figure()
            times = np.array(self.eval_times) / 365
            for i, (auc_d, auc_err_d) in enumerate(zip(self.auc_all, self.auc_err_all)):
                auc = auc_d[set_]
                auc_err = auc_err_d[set_]
                # plot
                label = self.labels[i]
                plt.errorbar(times + (i * 20 - 40) / 365, auc, yerr=auc_err, fmt='o', capsize=0, markersize=3,
                             label=label, color=self.colors[i])
                plt.xlabel("Years")
                plt.ylabel("Time-dependent AUC")
            plt.ylim(0.5, 0.85)
            plt.tight_layout()
            plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0., loc="upper left")
            filename = f"auc_all_{set_}.{self.plot_format}"
            plt.savefig(os.path.join(self.out_dir, filename), dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def km_cif_by_risk(self, scaler, selector, model, strategy, curve_type, n_groups=3):
        tab10_colors = cm.tab10.colors
        green = tab10_colors[2]  # Green in tab10
        red = tab10_colors[3]  # Red in tab10
        middle_colors = [c for i, c in enumerate(tab10_colors) if i not in [2, 3]][1:2]
        colors = [green] + middle_colors + [red] if n_groups > 2 else [green, red]
        for set_, risk_scores, y, comp_ev_label in zip(["val", "test"],
                                                       [self.risk_scores_val, self.risk_scores_test],
                                                       [self.y_val, self.y_test],
                                                       [self.competing_events_label_val, self.competing_events_label_test]):
            grid = np.quantile(risk_scores, np.linspace(start=0, stop=1, num=n_groups + 1))
            bins = np.digitize(risk_scores, grid[1:-1], right=True)
            subsets = [(bins == i) for i in range(n_groups)]  # Create boolean masks for each group
            labels = ["Low Risk", "Medium Risk", "High Risk"] if n_groups == 3 else ["Low Risk", "High Risk"]
            curves = []
            plt.figure()
            for subset, color, label in zip(subsets, colors, labels):
                if curve_type == "km":
                    curve = KaplanMeierFitter()
                    curve.fit(durations=y[self.time_column][subset] / 365,
                              event_observed=y[self.event_column][subset],
                              label=label)
                    curve.plot_survival_function(color=color, loc=slice(0, 1825 / 365), ci_show=False, linewidth=3)
                elif curve_type == "cif":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=Warning)
                        curve = AalenJohansenFitter()
                        curve.fit(durations=y[self.time_column][subset] / 365,
                                  event_observed=np.array(comp_ev_label[subset]),
                                  event_of_interest=1,
                                  label=label)
                        curve.plot_cumulative_density(color=color, loc=slice(0., 1826. / 365), at_risk_counts=False, linewidth=3)
                else:
                    logger.error(f"Survival/CIF plot: curve type {curve_type} not valid")
                    return
                curves.append(curve)
            _ = plt.legend(loc="best")
            y_label = 'Survival Probability' if curve_type == "km" else "Cumulative Incidence"
            filename_base = "km_by_risk" if curve_type == "km" else "cif_by_risk"
            plt.xlabel(xlabel='Years', fontsize=15)
            plt.ylabel(ylabel=y_label, fontsize=15)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            if curve_type == "cif":
                plt.ylim(-0.02, 0.38)
            add_at_risk_counts(*curves, rows_to_show=["At risk"])
            plt.tight_layout()
            filename = f"{filename_base}_{scaler}_{selector}_{model}_{strategy}_{set_}.{self.plot_format}"
            plt.savefig(os.path.join(self.out_dir, filename), dpi=self.dpi)
            plt.close()

    def calibration_plot(self, scaler, selector, model, search, train_strategy):
        calib_curves = {"val": {}, "test": {}}
        for set_, risk_scores, y in zip(["val", "test"],
                                        [self.risk_scores_times_val, self.risk_scores_times_test],
                                        [self.y_val, self.y_test]):
            for i, time_name in enumerate(self.eval_times_names):
                fig, grid, predict_grid, risk = calibration_plot_survival(durations=y[self.time_column],
                                                                          events=y[self.event_column],
                                                                          risk=risk_scores[:, i],
                                                                          time=self.eval_times[i])
                calib_curves[set_][time_name] = {
                    "grid": grid,
                    "predict_grid": predict_grid,
                    "risk": risk
                }
        return calib_curves

    def calibration_plot_all(self):
        xy_lim = 0.701
        for set_ in ["val", "test"]:
            for i, time_name in enumerate(self.eval_times_names):
                fig, ax = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
                ax1 = ax[0]
                ax1.plot([0, xy_lim], [0, xy_lim], color='black', linewidth=1)
                for j, calib_curve in enumerate(self.calib_curves):
                    calib_curve_t = calib_curve[set_][time_name]
                    ax1.plot(calib_curve_t["grid"], calib_curve_t["predict_grid"], "-", linewidth=2,
                             color=self.colors[j], label=self.labels[j])
                    # Density plot (separate subplot at bottom)
                    ax2 = ax[1]
                    sns.kdeplot(calib_curve_t["risk"], ax=ax2, color=self.colors[j], bw_adjust=0.5, fill=True, alpha=0.1)
                    ax2.set_ylabel("Density")
                    ax2.set_xlabel("Predicted probability")
                plt.xlim((-0.01, xy_lim))
                ax1.set_ylim((-0.01, xy_lim))
                ax1.set_ylabel("Observed probability")
                ax1.set_title(f"{time_name} calibration curve")
                filename = f"calibration_plot_all_{time_name}_{set_}.{self.plot_format}"
                ax1.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, filename), dpi=self.dpi)
                plt.close()

    def to_structured_array(self, df):
        return np.array(
            list(zip(df[self.event_column], df[self.time_column])),
            dtype=[(self.event_column, '?'), (self.time_column, '<f8')],
        )
