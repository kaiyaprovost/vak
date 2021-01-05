"""Utilities for input validation

adapted in part from scikit-learn under license
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/validation.py
"""
from pathlib import Path, PurePath
import warnings

import numpy as np


def column_or_1d(y, warn=False):
    """ravel column or 1d numpy array, else raise an error

    Parameters
    ----------
    y : array-like
    warn : boolean, default False
       To control display of warnings.

    Returns
    -------
    y : array
    """
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          stacklevel=2)
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))


def row_or_1d(y, warn=False):
    """ravel row or 1d numpy array, else raise an error

    Parameters
    ----------
    y : array-like
    warn : boolean, default False
       To control display of warnings.

    Returns
    -------
    y : array
    """
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[0] == 1:
        if warn:
            warnings.warn("A row-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          stacklevel=2)
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))


def is_a_directory(dir_path):
    """validate that ``dir_path`` is a directory.

    Handles strings that end in `/**`,
    which indicates "recursively search this directory",
    by removing that "suffix" and checking that
    what precedes it is an existing directory.

    Parameters
    ----------
    dir_path: str, pathlib.Path
    """
    if not (isinstance(dir_path, str) or isinstance(dir_path, PurePath)):
        raise TypeError(
            f'path must be a string or pathlib.Path, but type was {type(dir_path)}'
        )

    if isinstance(dir_path, PurePath):
        dir_path = str(dir_path)

    if dir_path.endswith('/**'):
        dir_to_validate = dir_path[:-2]  # remove terminal asterisks
    else:
        dir_to_validate = dir_path

    if not Path(dir_to_validate).exists():
        raise NotADirectoryError(
            f'path was not recognized as a directory: {dir_path}'
        )
    else:
        return True
