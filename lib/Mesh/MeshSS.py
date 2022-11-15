"""
author: xmaple
date: 2021-10-01
All Class for Single Step discritization methods, including Uniform_FO, Uniform_SO, Nonuniform_FO, Nonuniform_SO, and Runge-Kutta.
FO indicates "first order"
SO indicates "second order"
"""
import numpy as np
from scipy.integrate import odeint
from lib.utils.xfuncs import floatEqual
from lib.utils.xfuncs import RelativeError

"""
配点是动力学方程成立的离散点
cps (collocation points) are the points where the dynamic equation must be satisfied.
mps是所有离散点
mps (mesh points) are the discrete points
"""


class ZOH:
    def __init__(self, ncp=0, mps=None):
        """
        Zero order hold discretization
        :param ncp: number discrete intervals. discretization points number = num+1
        :param mps: mps location, belongs to [-1, 1]. the boundary points must be -1 and 1.
        """
        if mps is not None:  # initialize by assigning the mesh points
            assert mps[0] == -1. and mps[-1] == 1
            self.xtao = mps
        else:  # initialize by assigning the mesh points num
            self.xtao = np.linspace(-1, 1, ncp + 1)
        self.cps = self.xtao[1:]
        self.utao = self.xtao[:-1]
        self.ncp = ncp
        self.nx = len(self.xtao)
        self.nu = len(self.utao)
        self.cpState = np.arange(1, self.nx)
        self.cpControl = np.arange(self.nu)
        tmpWeights = np.concatenate([[-1], (self.xtao[1:] + self.xtao[:-1]) / 2, [1]])
        self.weights4State = tmpWeights[1:] - tmpWeights[:-1]
        self.weights4Control = self.xtao[1:] - self.xtao[:-1]


class FOH:
    def __init__(self, ncp=0, mps=None):
        """
        Zero order hold discretization
        :param ncp: number discrete intervals. discretization points number = num+1
        :param mps: mps location, belongs to [-1, 1]. the boundary points must be -1 and 1.
        """

        if mps is not None:  # initialize by assigning the mesh points
            assert mps[0] == -1. and mps[-1] == 1
            self.xtao = mps
        else:  # initialize by assigning the mesh points num
            self.xtao = np.linspace(-1, 1, ncp + 1)
        self.cps = self.xtao[1:]
        self.utao = self.xtao
        self.ncp = ncp
        self.nx = len(self.xtao)
        self.nu = len(self.utao)
        self.cpState = np.arange(1, self.nx)
        self.cpControl = np.arange(1, self.nx)

        tmpWeights = np.concatenate([[-1], (self.xtao[1:] + self.xtao[:-1]) / 2, [1]])
        self.weights4State = tmpWeights[1:] - tmpWeights[:-1]
        self.weights4Control = self.weights4State.copy()


class RungeKutta:
    def __init__(self, ncp=0, mps=None):
        """
        Zero order hold discretization
        :param ncp: number of discrete intervals. discretization points number = num + 1
        :param mps: mps location, belongs to [-1, 1]. the boundary points must be -1 and 1.
        """
        if mps is not None:  # initialize by assigning the mesh points
            assert mps[0] == -1. and mps[-1] == 1
            self.xtao = mps
        else:  # initialize by assigning the mesh points num
            self.xtao = np.linspace(-1, 1, ncp + 1)
        self.cps = self.xtao[1:]
        self.utao = self.xtao
        self.ncp = ncp
        self.nx = len(self.xtao)
        self.nu = len(self.utao)
        self.cpState = np.arange(1, self.nx)
        self.cpControl = np.arange(1, self.nx)

        tmpWeights = np.concatenate([[-1], (self.xtao[1:] + self.xtao[:-1]) / 2, [1]])
        self.weights4State = tmpWeights[1:] - tmpWeights[:-1]
        self.weights4Control = self.weights4State.copy()


class Trapezoidal:
    def __init__(self, ncp=0, mps=None):
        """
        Zero order hold discretization
        :param ncp: number of discrete intervals. discretization points number = num + 1
        :param mps: mps location, belongs to [-1, 1]. the boundary points must be -1 and 1.
        """
        if mps is not None:  # initialize by assigning the mesh points
            assert mps[0] == -1. and mps[-1] == 1
            self.xtao = mps
        else:  # initialize by assigning the mesh points num
            self.xtao = np.linspace(-1, 1, ncp + 1)
        self.cps = self.xtao[1:]
        self.utao = self.xtao
        self.ncp = ncp
        self.nx = len(self.xtao)
        self.nu = len(self.utao)
        self.cpState = np.arange(1, self.nx)
        self.cpControl = np.arange(1, self.nx)

        tmpWeights = np.concatenate([[-1], (self.xtao[1:] + self.xtao[:-1]) / 2, [1]])
        self.weights4State = tmpWeights[1:] - tmpWeights[:-1]
        self.weights4Control = self.weights4State.copy()


if __name__ == '__main__':
    rk = RungeKutta(ncp=10)
    print(rk.weights4State)
