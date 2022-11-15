"""
author: xmaple
date: 2021-10-21
"""
import numpy as np
import cvxpy as cvx
from typing import Union


def maxRelativeError(a, b):
    return np.max(RelativeError(a, b))


def PointwiseRelativeError(a, b):
    return np.abs(a - b) / (1. + np.max(np.abs(b)))


def RelativeError(a, b):
    if len(a.shape) == 1:
        return np.linalg.norm(np.abs(a - b) / (1. + np.max(np.abs(b))))
    else:
        return np.linalg.norm(np.abs(a - b) / (1. + np.max(np.abs(b))), axis=1)


def floatEqual(a, b):
    if abs(a - b) < 1e-12:  # threshold to determine whether two floats are equal
        return True
    else:
        return False


def npcvx_sin(x):
    if isinstance(x, np.ndarray) or isinstance(x, np.float):
        return np.sin(x)
    else:
        return np.sin(x.value)


def npcvx_cos(x):
    if isinstance(x, np.ndarray) or isinstance(x, np.float):
        return np.cos(x)
    else:
        return np.cos(x.value)


def npcvx_abs(x):
    if isinstance(x, np.ndarray) or isinstance(x, np.float):
        return np.abs(x)
    else:
        return cvx.abs(x)


def npcvx_power(x, p):
    if isinstance(x, np.ndarray) or isinstance(x, np.float):
        return np.power(x, p)
    else:
        return cvx.power(x, p)


def npcvx_norm(x, p: Union[int, str] = 2, axis=None):
    if isinstance(x, np.ndarray):
        return np.linalg.norm(x, ord=p, axis=axis)  # TODO
    else:
        return cvx.norm(x, p=p, axis=axis)


def npcvx_multiply(x, y):
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return x * y
    else:
        return cvx.multiply(x, y)
