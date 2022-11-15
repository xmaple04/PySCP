"""
author: xmaple
date: 2022-01-27
"""
import numpy as np
from lib.utils.plotlib import *

H0 = 7200
rho0 = 1.225
g = 9.8
Bc = 1e2


def getConst(h0, v0, zeta):
    return np.exp(h0 / H0) / v0 + rho0 / (4 * g * Bc) * (1 + zeta ** 2) * v0 ** 2


def h_of_v(v, d, zeta):
    h = H0 * np.log(d * v - rho0 / (4 * g * Bc) * (1 + zeta ** 2) * v ** 3)
    return h


zeta = 3
v = np.linspace(100, 3000, 1000)
d = getConst(100e3, 3000, zeta=zeta)
h = h_of_v(v, d, zeta)
plt.figure()
plt.plot(v, h)
plt.show()
