import numpy as np
from lib.Mesh.MeshPS import LegendreGauss, LegendreGaussRadau, flippedLegendreGaussRadau
from lib.Mesh.MeshSS import RungeKutta
from lib.utils.xfuncs import floatEqual


class Hybrid:
    def __init__(self, segDegrees, segNames, segFractions):
        assert len(segNames) == len(segFractions)
        assert len(segDegrees) == len(segFractions)
        assert floatEqual(sum(segFractions), 1)

        segDegrees = np.array(segDegrees).astype(int)  # convert to ndarray
        self.nseg = len(segDegrees)  # number of segments
        self.segDegrees = segDegrees  # degrees in each segment
        self.segFractions = segFractions  # fraction of each segment
        self.segBoundaryPoints = np.concatenate([[0], np.cumsum(segFractions)]) * 2 - 1
        self.segNames = segNames
        self.segMethods = []
        self.ncp = np.sum(segDegrees)
        self.nx = self.ncp + 1

        cps = []
        for i, sm in enumerate(segNames):
            tao0 = self.segBoundaryPoints[i]
            taof = self.segBoundaryPoints[i + 1]
            if sm == 'RK':
                method = RungeKutta(segDegrees[i])
                self.segMethods.append(method)
                cps.append(method.cps * (taof - tao0) / 2 + (taof + tao0) / 2)
            elif sm == 'fLGR':
                method = flippedLegendreGaussRadau(segDegrees[i])
                self.segMethods.append(method)
                cps.append(method.cps * (taof - tao0) / 2 + (taof + tao0) / 2)
            else:
                Exception('Waiting for development!..., please use RK and fLGR instead of {}.'.format(sm))

        self.cps = np.hstack(cps)
        self.xtao = np.concatenate([[-1], self.cps])
        if segNames[0] == 'RK':
            self.utao = np.concatenate([[-1], self.cps])
        else:
            self.utao = self.cps

        self.nu = len(self.utao)
        self.weights4State = np.zeros((self.nseg, self.nx))
        self.weights4Control = np.zeros_like(self.utao)

        self.PIMs = []
        self.PIMX0Indices = []
        self.PIMXfIndices = []
        leftBound = np.concatenate([[0], np.cumsum(self.segDegrees[:-1])])
        rightBound = np.cumsum(self.segDegrees)
        self.segStateIndex = [slice(leftBound[i], rightBound[i] + 1) for i in range(self.nseg)]
        self.segControlIndex = []
        self.cpState = np.arange(1, self.ncp + 1)
        for i in range(self.nseg):
            if segNames[i] == 'RK':
                if segNames[0] == 'RK':
                    self.segControlIndex.append(slice(leftBound[i], rightBound[i] + 1))
                else:
                    self.segControlIndex.append(slice(leftBound[i] - 1, rightBound[i]))
                self.PIMs.append(None)
                xt = self.xtao[self.segStateIndex[i]]
                tmpWeights = np.concatenate([[xt[0]], (xt[1:] + xt[:-1]) / 2, [xt[-1]]])
                self.weights4State[i, self.segStateIndex[i]] = tmpWeights[1:] - tmpWeights[:-1]
                self.PIMX0Indices.append(None)
                self.PIMXfIndices.append(None)
            else:
                if segNames[0] == 'RK':
                    self.segControlIndex.append(slice(leftBound[i] + 1, rightBound[i] + 1))
                else:
                    self.segControlIndex.append(slice(leftBound[i], rightBound[i]))
                self.PIMs.append(self.segMethods[i].PIM * self.segFractions[i])  # PIM multiply
                self.weights4State[i, self.segStateIndex[i]] = self.segMethods[i].integralWeights * self.segFractions[i]
                self.PIMX0Indices.append([0] * self.segDegrees[i])
                self.PIMXfIndices.append(np.arange(1, self.segDegrees[i] + 1))
        self.weights4State = np.squeeze(np.sum(self.weights4State, axis=0))

        if segNames[0] == 'RK':
            self.weights4Control = self.weights4State
        else:
            self.weights4Control = self.weights4State[1:]


if __name__ == '__main__':
    disc = Hybrid(segDegrees=[10, 10], segNames=['RK', 'fLGR'], segFractions=[1 / 2] * 2)
    disc = Hybrid(segDegrees=[10, 10], segNames=['fLGR', 'RK'], segFractions=[1 / 2] * 2)
    print(disc.xtao.shape)
    print(disc.utao.shape)
    print(disc.segStateIndex)
    print(disc.segControlIndex)
    # from lib.utils.plotlib import *
    # plt.figure()
    # plt.plot(disc.cps, np.ones_like(disc.cps), marker='x')
    # plt.plot(disc.mps, np.ones_like(disc.mps), marker='o')
    # plt.show()
