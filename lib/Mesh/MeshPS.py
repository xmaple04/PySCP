"""
Author: xmaple
Date:   2021-09-27
PIM:    pseudo-spectral integral matrix
PDM:    pseudo-spectral differential matrix
cps:    collocation points.
xtao:  points used in Lagrangian interpolation.
"""

import numpy as np
from scipy import special


class LegendreGauss:
    def __init__(self, nc):
        self.ncp = nc
        self.nx = nc + 2
        self.cps = self.generate_collocation_points()
        self.xtao = np.concatenate([[-1], self.cps, [1]])
        self.utao = self.cps

        self.jacobiWeights = self.generate_jacobi_weights()
        self.integralWeights = np.concatenate([[0], self.jacobiWeights, [0]])
        self.PDM, self.PIM = self.generate_PIM_PDM()

    def generate_collocation_points(self):
        roots, _ = special.roots_jacobi(self.ncp, 0, 0)
        return roots

    def generate_jacobi_weights(self):
        """ Return Gauss-Legendre weights. """
        w = np.zeros((self.ncp,))
        for i in range(self.ncp):
            _, derivative = special.lpn(self.ncp, self.cps[i])
            w[i] = 2 / ((1 - self.cps[i] ** 2) * derivative[-1] ** 2)
        return w

    def barycentricWeights(self):
        ksi = (-1) ** np.arange(self.ncp) * np.sqrt((1 - self.cps ** 2) * self.jacobiWeights)
        return ksi

    def generate_PIM_PDM(self):
        ksi = self.barycentricWeights()

        D = np.zeros((self.ncp, self.ncp))
        for k in range(self.ncp):
            for i in range(self.ncp):
                if k != i:
                    D[k, i] = ksi[i] / ksi[k] / (self.cps[k] - self.cps[i])
        for k in range(self.ncp):
            D[k, k] = -np.sum(D[k, :])

        PDM = np.zeros((self.ncp, self.nx))
        for k in range(self.ncp):
            for i in range(self.ncp):
                if k == i:
                    PDM[k, i + 1] = (1 + (self.cps[k] + 1) * D[k, i]) / (self.cps[i] + 1)
                else:
                    PDM[k, i + 1] = ((self.cps[k] + 1) * D[k, i]) / (self.cps[i] + 1)
        PDM[:, 0] = -np.sum(PDM[:, 1:], axis=1)  # 其余元素的和的相反数
        PIM = np.zeros((self.ncp + 1, self.ncp))
        PIM[:-1] = np.linalg.inv(PDM[:, 1:-1])
        PIM[-1] = self.integralWeights[1:-1]
        return PDM, PIM


class LegendreGaussRadau:
    """ Legendre-Gauss-Radau Pseudospectral method
    Gauss-Radau xtao are roots of :math:`P_n(x) + P_{n-1}(x)`. """

    def __init__(self, nc):
        self.ncp = nc
        self.nx = nc + 1
        self.cps = self.generate_collocation_points()
        self.xtao = np.concatenate([self.cps, [1]])

        self.jacobiWeights = self.generate_jacobi_weights()
        self.integralWeights = np.concatenate([self.jacobiWeights, [0]])
        self.PDM, self.PIM = self.generate_PIM_PDM()

    def generate_collocation_points(self):
        """ Return Gauss-Radau-Radau collocation points. """
        roots, _ = special.roots_jacobi(self.ncp - 1, 0, 1)
        cps = np.hstack((-1, roots))
        return cps

    def generate_jacobi_weights(self):
        """ Return Gauss-Legendre-Radau weights. """
        tao = self.cps
        w = np.zeros((self.ncp,))
        for i in range(self.ncp):
            pnz, _ = special.lpn(self.ncp - 1, self.cps[i])
            w[i] = (1 - tao[i]) / (self.ncp ** 2 * pnz[-1] ** 2)
        return w

    def generate_PIM_PDM(self):
        Y = np.tile(self.xtao.reshape((-1, 1)), (1, self.nx))
        Ydiff = Y - Y.transpose() + np.eye(self.nx)
        WW = np.tile((1. / np.prod(Ydiff, axis=0)).reshape((-1, 1)), (1, self.nx))
        D = WW / (WW.transpose() * Ydiff)
        np.fill_diagonal(D, 1 - sum(D))  # 替换对角元素

        # full differentiation matrix
        D = -D.transpose()
        PDM = D[:-1, :]

        # find integration matrix E
        PIM = np.linalg.inv(PDM[:, 1: self.nx])
        return PDM, PIM

    def barycentricWeights(self):
        ksi = (-1) ** np.arange(self.ncp) * np.sqrt((1 - self.cps) * self.integralWeights)
        ksi[0] = 2 * np.sqrt(self.jacobiWeights[0])
        return ksi


class flippedLegendreGaussRadau:
    """ flipped Legendre-Gauss-Radau Pseudospectral method
    Gauss-Radau xtao are roots of :math:`P_n(x) - P_{n-1}(x)`.
    Note that fLGR does not have weights"""

    def __init__(self, nc):
        self.ncp = nc
        self.nx = nc + 1
        self.cps = self.generate_collocation_points()
        self.xtao = np.concatenate([[-1], self.cps])

        self.jacobiWeights = self.generate_jacobi_weights()
        self.integralWeights = np.concatenate([[0], self.jacobiWeights])
        self.PDM, self.PIM = self.generate_PIM_PDM()

    def generate_collocation_points(self):
        """ Return flipped-Gauss-Legendre-Radau collocations points. """
        roots, _ = special.roots_jacobi(self.ncp - 1, 0, 1)
        cps = np.hstack((-1, roots))
        cps = -cps[::-1]
        return cps

    def generate_jacobi_weights(self):
        """Return flipped-Gauss-Legendre-Radau weight."""
        tao = self.cps
        w = np.zeros((self.ncp,))
        for i in range(self.ncp):
            pnz, _ = special.lpn(self.ncp, -self.cps[i])
            w[i] = (1 + tao[i]) / (self.ncp ** 2 * pnz[-1] ** 2)
        return w

    def generate_PIM_PDM(self):
        tao = self.cps
        ksi = np.zeros((self.ncp,))
        for i in range(self.ncp):
            ksi[i] = (-1) ** i * np.sqrt((1 + tao[i]) * self.jacobiWeights[i])

        D = np.zeros((self.ncp, self.ncp))
        for k in range(self.ncp):
            for i in range(self.ncp):
                if k != i:
                    D[k, i] = ksi[i] / ksi[k] / (tao[k] - tao[i])
        for k in range(self.ncp):
            D[k, k] = -np.sum(D[k, :])

        PDMif = np.zeros((self.ncp, self.nx))
        for k in range(self.ncp):
            for i in range(self.ncp):
                if k == i:
                    PDMif[k, i + 1] = (1 + (tao[k] + 1) * D[k, i]) / (tao[i] + 1)
                else:
                    PDMif[k, i + 1] = ((tao[k] + 1) * D[k, i]) / (tao[i] + 1)
        PDMif[:, 0] = -np.sum(PDMif[:, 1:], axis=1)  # 其余元素的和的相反数

        PIMnt = np.linalg.inv(PDMif[:, 1: self.nx])
        return PDMif, PIMnt


class Legendre_Gauss_Lobatto:
    """ TODO """

    def __init__(self, nc):
        self.ncp = nc
        self.cps = self.generate_collocation_points()
        self.weights = self.generate_jacobi_weights()

    def generate_collocation_points(self):
        """ Legendre-Gauss-Lobatto(LGL) points"""
        roots, weight = special.roots_jacobi(self.ncp - 2, 1, 1)
        xtao = np.hstack((-1, roots, 1))
        return xtao

    def generate_jacobi_weights(self):
        """ Legendre-Gauss-Lobatto(LGL) weights."""
        w = np.zeros_like(self.cps)
        for i in range(self.ncp):
            Pn, _ = special.lpn(self.ncp - 1, self.cps[i])
            w[i] = 2 / (self.ncp * (self.ncp - 1) * Pn[-1] ** 2)
        return w


class ChebyshevPseudo:
    """ TODO """

    def __init__(self, nc):
        super().__init__('Chebyshev', nc)

    def generate_collocation_points(self):
        tao = -np.cos(np.arange(self.ncp) / self.ncp * np.pi)
        return tao

    def generate_PIM_PDM(self):
        return 0


if __name__ == '__main__':
    import time

    num = 100
    start = time.time()
    for i in range(num):
        ps = LegendreGauss(30)
    end = time.time()
    print('时间:{}ms'.format((end - start) / num * 1000))
