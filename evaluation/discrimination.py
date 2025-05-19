import numpy as np
import pandas as pd
from pycox.evaluation.eval_surv import EvalSurv


def antolini_concordance_index(durations, labels, cuts, risk, time_max):
    """Compute time-dependent concordance index (Antolini et al.), up to given time.
    :param durations: Event/censoring times
    :param labels: array of event indicators (0: censored, 1:event of interest)
    :param risk: risk predicted at each time point of "grid"
    :param cuts: grid time points
    :param time_max: maximum time to consider for the computation of concordance index
    :return: time-dependent c-index
    """
    max_i_grid = np.nonzero(cuts <= time_max)[0].max()  # max index in the grid corresponding to "time_max"
    surv = 1 - risk[:, 0:max_i_grid]  # survival until time_max
    surv_df = pd.DataFrame(surv.transpose(), cuts[0:max_i_grid])
    ev = EvalSurv(surv_df, durations, labels, censor_surv='km')  # Changing "censor_surv" does not change anything
    c_index = ev.concordance_td('adj_antolini')
    return c_index

