import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from lifelines import KaplanMeierFitter
from rpy2.robjects import r
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from cmprsk.rpy_utils import r_vector
pandas2ri.activate()


def mean_calibration(durations, events, risk_scores, eval_times):
    """
    Mean calibration (or “calibration-in-the-large”)measures agreement of the predicted and observed survival fraction
    in the external validation data set. It indicates systematic under-prediction or over-prediction. Fixed time point
    mean calibration is the ratio of the observed incidence and the average predicted risk.
    """
    km = KaplanMeierFitter()
    km.fit(durations=durations, event_observed=events)
    incidence = km.cumulative_density_at_times(eval_times)
    mean_predicted_risk = np.mean(risk_scores, axis=0)
    mean_calibration = incidence / mean_predicted_risk
    return mean_calibration


def ici_survival(durations, labels, risk, time, return_calib_model=False):
    """Function to compute the integrated calibration index for time-to-event outcome, at a given time instant.
    To produce smooth calibration curves, the hazard of the outcome is regressed on the predicted outcome risk using a
    flexible regression model. Then the ICI is the weighted difference between smoothed observed proportions and
    predicted risks.

    Reference: Austin et al. (2020). https://doi.org/10.1002/sim.8570

    Input
     - durations {np array}: Event/censoring times
     - labels: array of event indicators
     - predictions: predicted risk at the given time
     - time: time instant at which ICI has to be computed
     - return_calib_model: bool, if True also the calibration model is returned
    Output
     - ici: integrated calibration index for survival outcome
     - calibrate: calibration model (optional)
    """
    risk = risk.astype("float64")
    risk[risk < 1e-15] = 1e-15  # discard too small values (because when I take the log I get Inf)
    risk[risk == 1] = 0.9999
    risk_cll = np.log(-np.log(1 - risk))
    risk_cll = np.clip(risk_cll, -10, 10)
    risk_cll = robjects.FloatVector(risk_cll)  # complementary log-log transformation
    polspline = importr("polspline")
    calibrate = polspline.hare(data=durations, delta=labels, cov=risk_cll, maxdim=5, penalty=3)
    predict_calibrate = np.array(polspline.phare(time, risk_cll, calibrate))
    ici = np.mean(np.abs(risk - predict_calibrate.squeeze()))  # Compute ici
    if return_calib_model:
        return ici, calibrate
    else:
        return ici


def ici_survival_times(durations, events, risk_times, times, return_calib_models=False, parallel=True):
    """Compute the ICI for time-to-event outcome at different time points, with or without competing risks.

        Input
         - durations: Events/censoring times
         - events: array of event indicators (0: censored, 1:event of interest, 2.... competing events)
         - risk_times: predicted risk at the given time points
         - time: time instants at which ICI has to be computed
         - return_calib_models: bool, if True also the calibration models are returned
         - parallel: bool, if True the ICI at different times are computed in parallel
        Output
         - ici: integrated calibration index for survival outcome at the given times
         - calibration_models: list with fitted calibration models (optional)
        """
    if parallel:
        ici_models = Parallel(n_jobs=30)(delayed(ici_survival)(durations, events, risk_times[:, i],
                                                               time, return_calib_model=True)
                                         for i, time in enumerate(times))
    else:
        ici_models = [ici_survival(durations, events, risk_times[:, i], t, return_calib_model=True)
                      for i, t in enumerate(times)]
    ici_at_times, calibration_models = zip(*ici_models)
    ici_at_times = np.array(ici_at_times)
    calibration_models = list(calibration_models)
    return (ici_at_times, calibration_models) if return_calib_models else ici_at_times


def calibration_plot_survival(durations, events, risk, time, calibration_model=None):
    """
    Plot survival calibration curve and risk distribution at a given time.

    Fits or uses a calibration model to estimate observed probabilities from predicted risks,
    and plots the calibration curve alongside the risk density.

    Parameters:
    - durations: array-like, survival times.
    - events: array-like, event indicators.
    - risk: array-like, predicted risk scores.
    - time: float, evaluation time point.
    - calibration_model: optional, pre-fitted calibration model.
    - comp_risk: bool, unused here (placeholder).
    - method: str, method for calibration model fitting (default "cox").

    Returns:
    - fig: matplotlib Figure object.
    - grid: points at which calibration curve is evaluated.
    - predict_grid: calibrated observed probabilities.
    - risk: input risk scores.
    """
    stats = importr("stats")
    labels = robjects.FloatVector(events)
    durations = robjects.FloatVector(durations)
    # Fitting the calibration model
    risk_r = r_vector(risk)
    grid = r.seq(stats.quantile(risk_r, probs=0.025), stats.quantile(risk_r, probs=0.975), length=100)
    grid_cll = robjects.FloatVector(np.log(-np.log(1 - np.array(grid))))  # complementary log-log grid
    if calibration_model is None:  # if no model is given as input, we fit it
        _, calibration_model = ici_survival(durations, labels, risk, time, return_calib_model=True)
    # Predicted probability for grid points
    polspline = importr("polspline")
    predict_grid = polspline.phare(time, grid_cll, calibration_model)
    # --- PLOT ---
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1 = ax[0]
    ax1.plot(grid, predict_grid, "-", linewidth=2, color="red")
    ax1.set_xlim((-0.02, max(grid) + 0.1))
    ax1.set_ylim((0, max(grid) + 0.1))
    ax1.set_ylabel("Observed probability")
    ax1.set_title(f"{int(time / 365)}-year calibration curve")
    ax1.plot([0, 1], [0, 1], color='black')
    # Density plot (separate subplot at bottom)
    ax2 = ax[1]
    sns.histplot(risk, ax=ax2, color='tab:blue')
    ax2.set_ylabel("Density")
    ax2.set_xlabel("Predicted probability")
    fig.tight_layout()  # Adjust layout
    plt.close()
    return fig, grid, predict_grid, risk
