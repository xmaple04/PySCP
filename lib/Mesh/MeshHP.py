import warnings

import numpy as np
from lib.Mesh.MeshPS import LegendreGauss, LegendreGaussRadau, flippedLegendreGaussRadau

"""
xtao: 离散点
nx: 离散点个数
cps: 配点，
ncp: 配点个数
degrees: 每个分段内的多项式阶数 [数组]
segFractions: 每个分段占整个归一化时域[-1, 1]的百分比
segLegendre: 每个分段的勒让德离散方法，对应一个MeshPS对象
segBoundaryPoints: 每个分段
cpState, cpControl: 循环配点时，状态变量和控制变量的索引
nx, nu: 状态变量和控制变量个数。
PDM, PIM: 微分与积分矩阵
PIMX0Index, PIMXfIndex: 各个配点在积分形式下的起始点与
weights4State, weights4Control: 离散积分型目标函数

"""


class hpPseudoSpectral:
    def __init__(self, degree, segFractions):
        """
         generalized hp pseudospectral method
        :param degree: list of polynomial degrees in each interval
        :param segFractions: ratio between each interval to the whole time domain length
        also known as interval, subinterval or subdomain.
        """
        degree = np.array(degree).astype(int)  # convert to ndarray
        self.ncp = np.sum(degree)
        self.nu = self.ncp

        self.nseg = len(degree)  # number of segments
        self.segDegrees = degree  # degrees in each segment
        self.segFractions = segFractions  # fraction of each segment

        # 状态变量和控制变量在离散点数组中的索引
        self.cpState = np.zeros((self.ncp,), dtype=int)
        self.nx = None
        self.xtao = None
        self.utao = None
        self.PDM = None
        self.PIM = None
        self.cps = None
        self.segLegendre = None
        self.weights4State = None
        self.weights4Control = None
        self.cpControl = np.arange(self.ncp)

    def setup(self, deg1, lg=False):
        cps = []
        self.xtao = np.zeros((self.nx,))
        self.utao = np.zeros((self.nu,))
        self.PDM = np.zeros((self.ncp, self.nx))
        self.PIM = np.zeros((self.nx - 1, self.ncp))

        self.weights4State = np.zeros((self.nseg, self.nx))
        segBoundaryPoints = np.concatenate([[0], np.cumsum(self.segFractions)]) * 2 - 1
        for i in range(self.nseg):
            tao0 = segBoundaryPoints[i]
            taof = segBoundaryPoints[i + 1]
            cps.append(self.segLegendre[i].cps * (taof - tao0) / 2 + (taof + tao0) / 2)
            segx_in_xs = slice(np.sum(deg1[:i]), np.sum(deg1[:i + 1]) + 1)  # segment meshpoints in the whole meshpoints
            segcps_in_cps = slice(np.sum(self.segDegrees[:i]), np.sum(self.segDegrees[:i + 1]))  # segment cps in the whole cps
            seg_mps_in_integrated = slice(np.sum(deg1[:i]), np.sum(deg1[:i + 1]))  # segment integrated points in all integrated points

            self.xtao[segx_in_xs] = self.segLegendre[i].xtao * (taof - tao0) / 2 + (taof + tao0) / 2
            self.weights4State[i, segx_in_xs] = self.segLegendre[i].integralWeights * self.segFractions[i]
            self.PDM[segcps_in_cps, segx_in_xs] = self.segLegendre[i].PDM / self.segFractions[i]  # PDM divide
            if lg:
                self.PIM[seg_mps_in_integrated, segcps_in_cps] = self.segLegendre[i].PIM * self.segFractions[i]  # PIM multiply
            else:
                self.PIM[segcps_in_cps, segcps_in_cps] = self.segLegendre[i].PIM * self.segFractions[i]  # PIM multiply

        self.cps = np.hstack(cps)
        self.utao = self.cps
        self.weights4State = np.squeeze(np.sum(self.weights4State, axis=0))
        self.weights4Control = self.weights4State[self.cpState]


class hpLG(hpPseudoSpectral):
    def __init__(self, degree, segFractions):
        super().__init__(degree=degree, segFractions=segFractions)
        self.nx = self.ncp + self.nseg + 1
        self.segLegendre = [LegendreGauss(d) for d in degree]

        deg1 = self.segDegrees + 1
        leftBound = np.concatenate([[0], np.cumsum(deg1[:-1])])
        self.PIMX0Index = np.hstack([[leftBound[i]] * (self.segDegrees[i] + 1) for i in range(self.nseg)])
        self.PIMXfIndex = np.hstack([leftBound[i] + np.arange(1, self.segDegrees[i] + 2) for i in range(self.nseg)])

        for i in range(self.nseg):
            seg_cps_in_cps = slice(np.sum(self.segDegrees[:i]), np.sum(self.segDegrees[:i + 1]))  # segment cps in the whole cps
            self.cpState[seg_cps_in_cps] = np.arange(np.sum(deg1[:i]) + 1, np.sum(deg1[:i + 1]))
        super().setup(deg1, lg=True)


class hpLGR(hpPseudoSpectral):
    def __init__(self, degree, segFractions):
        super().__init__(degree=degree, segFractions=segFractions)
        self.nx = self.ncp + 1
        self.nx = self.nx
        self.segLegendre = [LegendreGaussRadau(d) for d in degree]

        leftBound = np.concatenate([[0], np.cumsum(self.segDegrees[:-1])])
        self.PIMX0Index = np.hstack([[leftBound[i]] * (self.segDegrees[i]) for i in range(self.nseg)])
        self.PIMXfIndex = np.hstack([leftBound[i] + np.arange(1, self.segDegrees[i] + 1) for i in range(self.nseg)])

        for i in range(self.nseg):
            seg_cps_in_cps = slice(np.sum(self.segDegrees[:i]), np.sum(self.segDegrees[:i + 1]))  # segment cps in the whole cps
            self.cpState[seg_cps_in_cps] = np.arange(np.sum(self.segDegrees[:i]), np.sum(self.segDegrees[:i + 1]))
        super().setup(self.segDegrees)


class hpfLGR(hpPseudoSpectral):
    def __init__(self, degree, segFractions):
        super().__init__(degree=degree, segFractions=segFractions)
        self.nx = self.ncp + 1
        self.nx = self.nx
        self.segLegendre = [flippedLegendreGaussRadau(d) for d in degree]

        leftBound = np.concatenate([[0], np.cumsum(self.segDegrees[:-1])])
        self.PIMX0Index = np.hstack([[leftBound[i]] * (self.segDegrees[i]) for i in range(self.nseg)])
        self.PIMXfIndex = np.hstack([leftBound[i] + np.arange(1, self.segDegrees[i] + 1) for i in range(self.nseg)])

        for i in range(self.nseg):
            seg_cps_in_cps = slice(np.sum(self.segDegrees[:i]), np.sum(self.segDegrees[:i + 1]))  # segment cps in the whole cps
            self.cpState[seg_cps_in_cps] = np.arange(np.sum(self.segDegrees[:i]), np.sum(self.segDegrees[:i + 1])) + 1
        super().setup(self.segDegrees)


if __name__ == '__main__':
    hplg = hpLG([6, 6, 6], segFractions=[1 / 3] * 3)
    hplgr = hpLGR([6, 6, 6], segFractions=[1 / 3] * 3)
    hpflgr = hpfLGR([6, 6, 6], segFractions=[1 / 3] * 3)
    print(hplg.PDM.shape, hplg.PIM.shape, hplg.cpState)
    print(hplgr.PDM.shape, hplgr.PIM.shape, hplgr.cpState)
    print(hpflgr.PDM.shape, hpflgr.PIM.shape, hpflgr.cpState)
    print(np.sum(hplg.weights4State))

    lg = LegendreGauss(6)
    lgr = LegendreGaussRadau(6)
    flgr = flippedLegendreGaussRadau(6)
    print(lg.PDM.shape, lg.PIM.shape)
    print(lgr.PDM.shape, lgr.PIM.shape)
    print(flgr.PDM.shape, flgr.PIM.shape)
