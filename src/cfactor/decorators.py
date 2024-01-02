import numpy as np


def check_nan(func):
    def wrapper(self, *args, **kwargs):
        for arg in args:
            if np.any(np.isnan(arg)):
                msg = "NaN detected in input"
                raise ValueError(msg)
        return func(self, *args, **kwargs)

    return wrapper
