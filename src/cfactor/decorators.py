import numpy as np


def check_nan(func):
    """
    This function can be used as a decorator to check if none of the inputs of a
    function contains NaN values.
    """

    def wrapper(self, *args, **kwargs):
        for arg in args:
            if np.any(np.isnan(arg)):
                msg = "NaN detected in input"
                raise ValueError(msg)
        return func(self, *args, **kwargs)

    return wrapper


def check_length(func):
    """
    This function can be used as a decorator to check if all the inputs of a function
    have the same dimensions.
    """

    def wrapper(self, *args, **kwargs):
        for i, arg in enumerate(args):
            if i == 0:
                shape = np.asarray(arg).shape
            elif shape != np.asarray(arg).shape:
                raise ValueError("Dimension mismatch for inputs")
        return func(self, *args, **kwargs)

    return wrapper
