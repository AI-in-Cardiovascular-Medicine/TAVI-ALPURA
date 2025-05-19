import numpy as np
from sksurv.functions import StepFunction


def _array_to_step_function(time_points, array):
    n_samples = array.shape[0]
    funcs = np.empty(n_samples, dtype=np.object_)
    for i in range(n_samples):
        funcs[i] = StepFunction(x=time_points, y=array[i])
    return funcs
