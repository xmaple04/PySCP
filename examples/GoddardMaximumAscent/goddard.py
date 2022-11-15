"""
author: xmaple
date: 2021-11-18
"""
import numpy as np
from lib.core.Entrance import PySCP
from lib.utils.PlanetData import Earth
from lib.utils.Scale import Scale
from lib.utils.Structs import PhaseInfo, Linkage


class moonLander:
    def __init__(self):
        self.ve = 3000

        # scale
        scale = Scale(np.array([1, 1, 1]),
                      np.array([1]),
                      1)
        ascent = PhaseInfo()
        ascent.init_state_bound = np.array([0, 0, 3], [0, 0, 3])
        ascent.state_bound = np.array([[0, -5, 1], [10, 10, 3]])
        ascent.final_state_bound = np.array([[0, -1, 1], [10, 10, 1]])
        ascent.control_bound = np.array([[0], [193]])
        ascent.t0_bound = np.array([0, 0])
        ascent.tf_bound = np.array([30, 60])
        ascent.dynamicsFunc = self.scpDynamics
        ascent.approx = self.scpApprox
        ascent.scale = scale
        ascent.trState = np.ones((3,))
        ascent.trControl = np.ones((1,))
        ascent.trSigma = 1
        self.phases = [ascent]
        self.linkages = []

    def objective(self, phases):
        return -phases[0].dv.state[-1, 0]

    def scpDynamics(self, x, u, t, auxdata):
        h, v, m = x
        TT = u[0]
        drag = 5.5e-5 * v ** 2 * np.exp(-h / 23800)

        f = np.zeros_like(x)
        f[0] = v
        f[1] = (TT - drag) / m - Earth.g0
        f[2] = -TT / self.ve
        return f

    def scpApprox(self, x, u, t, auxdata):
        h, d, v, gamma, m = x
        Tx, Ty, TT = u
        xdim = 3
        udim = 1

        A = np.zeros((xdim, xdim))
        A[0, 2] = np.sin(gamma)
        A[0, 3] = v * np.cos(gamma)
        A[1, 0] = -v * np.cos(gamma) * planet.radius / r ** 2
        A[1, 2] = np.cos(gamma) * planet.radius / r
        A[1, 3] = -v * np.sin(gamma) * planet.radius / r

        A[2, 0] = 2 * planet.mu / (r ** 3) * np.sin(gamma)
        A[2, 3] = -planet.mu / (r ** 2) * np.cos(gamma)
        A[2, 4] = - tp * Tx / (m ** 2)
        A[3, 0] = np.cos(gamma) * (-v / (r ** 2) + 2 * planet.mu / ((r ** 3) * v))
        A[3, 2] = np.cos(gamma) * (1 / r + planet.mu / ((r ** 2) * (v ** 2))) - Ty * tp / (m * (v ** 2))
        A[3, 3] = -np.sin(gamma) * (v / r - planet.mu / ((r ** 2) * v))
        A[3, 4] = - tp * Ty / ((m ** 2) * v)

        B = np.zeros((xdim, udim))
        B[2, 0] = tp / m
        B[3, 1] = tp / (m * v)
        B[4, 2] = -1 / (planet.g0 * self.Isp[stage - 1])
        return A, B


if __name__ == '__main__':
    vehicle = moonLander()

    meshConfig = {'discrete': 'fLGR',
                  'scheme': 'differential',  # difference or integration
                  'adaptive': False,
                  'degree': [10] * 5,
                  'seg_fractions': [0.2] * 5,
                  'tolerance': 1e-6,
                  'maxDegree': 50,
                  'maxIteration': 30}

    setup = {'model': vehicle,
             'meshConfig': meshConfig,
             'algorithm': 'TrustRegion',
             'initialization': 'linear',
             'verbose': 1}

    prob = PySCP(setup)
    prob.solve()
    prob.plotXU()
    prob.print()
