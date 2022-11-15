"""
author: xmaple
date: 2021-11-08
"""
import numpy as np


class IterativeMethod:
    def __init__(self, method, setup):
        self.method = method
        self.phaseNum = 1
        self.nonMonotone = 1  # 非单调信赖域法中该参数不等于1
        self.beta = 0.3
        self.weightDynamics = [3e3]
        self.weightPath = [1e2]
        self.weightBoundary = [1e2]
        self.weightLinkage = 1e2

        if setup.get('beta'):
            self.beta = setup.get('beta')
        if setup.get('weightNu'):
            self.weightDynamics = setup.get('weightNu')
        if setup.get('weightPath'):
            self.weightPath = setup.get('weightPath')
        if setup.get('weightBoundary'):
            self.weightBoundary = setup.get('weightBoundary')
        if setup.get('weightLinkage'):
            self.weightLinkage = setup.get('weightLinkage')

        self.JTrace = []
        self.fakeJTrace = []
        self.allJTrace = []
        self.fakeAllJTrace = []

    def isTrustRegion(self):
        if self.method == 'trust-region':
            return True
        else:
            return False

    def setParams(self, setup):
        for key, value in setup.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def converged(self):
        if len(self.JTrace) < 2:
            return False
        elif np.abs(self.JTrace[-1] - self.JTrace[-2]) < 1e-6:
            return True


class TrustRegion(IterativeMethod):
    def __init__(self, setup):
        super().__init__('trust-region', setup)
        # default values
        self.thresh_reject = 0.04
        self.thresh_shrink = 0.15
        self.thresh_expand = 0.75
        self.ratio_shrink = 0.5
        self.ratio_expand = 3.2

        self.trState = []
        self.trControl = []
        self.trSigma = []
        self.ratioHistory = []
        self.ratios = [1, 1]

        # user define
        self.setParams(setup)

    def initJTrace(self, J):
        self.JTrace.append(J)
        self.allJTrace.append(J)

    def TrustRegionAdaption(self, J):
        delta_J = self.JTrace[-1] - J
        self.allJTrace.append(J)
        self.JTrace.append(J)

        if len(self.JTrace) > 2:
            r = (1 - np.abs((self.JTrace[-1] - self.JTrace[-2]) / self.JTrace[-2])) * self.ratios[-1]
            self.ratios.append(r)
        if delta_J < 0:
            self.adaptTrustRegion(self.ratio_shrink)
            return False
        return True

    def adaptTrustRegion(self, ratio):
        self.ratioHistory.append(ratio)
        for iPhase in range(self.phaseNum):
            self.trState[iPhase] = self.trState[iPhase] * ratio
            # self.trControl[iPhase] = self.trControl[iPhase] * ratio
            self.trSigma[iPhase] = self.trSigma[iPhase] * ratio

    def setupTrustRegion(self, phases):
        self.phaseNum = len(phases)
        """ different phases may have different state variables or control variables, so the trust region must be list instead of ndarray """
        for iPhase, phase in enumerate(phases):
            self.trState.append(phase.trState.reshape((1, -1)))
            self.trControl.append(phase.trControl.reshape((1, -1)))
            self.trSigma.append(phase.trSigma)

    def initCost(self, realJ, fakeJ):
        self.JTrace.append(realJ)
        self.fakeJTrace.append(fakeJ)


class TrustRegionLineSearch(IterativeMethod):
    def __init__(self, setup):
        super().__init__('trust-region-line-search', setup)
        # default values
        self.thresh_reject = 0.04
        self.thresh_shrink = 0.15
        self.thresh_expand = 0.75
        self.ratio_shrink = 0.5
        self.ratio_expand = 3.2

        self.trState = []
        self.trControl = []
        self.trSigma = []
        self.ratioHistory = []

        # user define
        self.setParams(setup)

    def TrustRegionAdaption(self, realJ, fakeJ):
        if len(self.JTrace) == 0:
            self.JTrace.append(realJ)
            self.fakeJTrace.append(fakeJ)
            self.allJTrace.append(realJ)
            self.fakeAllJTrace.append(fakeJ)
            return True

        delta_J = self.JTrace[-1] - realJ
        delta_L = self.JTrace[-1] - fakeJ
        self.allJTrace.append(realJ)
        self.fakeAllJTrace.append(fakeJ)

        rho = delta_J / delta_L
        if rho < self.thresh_reject or delta_J < 0:
            print('reject, shrink', self.trSigma)
            # 拒绝该结果
            self.adaptTrustRegion(self.ratio_shrink)
            return False
        else:
            self.JTrace.append(realJ)
            self.fakeJTrace.append(fakeJ)
            if rho >= self.thresh_expand:
                self.adaptTrustRegion(self.ratio_expand)
                print('accept, {:.6f}, expand'.format(rho))
            elif rho < self.thresh_shrink:
                self.adaptTrustRegion(self.ratio_shrink)
                print('accept, {:.6f}, shrink'.format(rho))
            else:
                print('accept, {:.6f}, unchanged'.format(rho))
            return True

    def adaptTrustRegion(self, ratio):
        self.ratioHistory.append(ratio)
        for iPhase in range(self.phaseNum):
            # self.trState[iPhase] = self.trState[iPhase] * ratio
            # self.trControl[iPhase] = self.trControl[iPhase] * ratio
            self.trSigma[iPhase] = self.trSigma[iPhase] * ratio

    def setupTrustRegion(self, phases):
        self.phaseNum = len(phases)
        """ different phases may have different state variables or control variables, so the trust region must be list instead of ndarray """
        for iPhase, phase in enumerate(phases):
            self.trState.append(phase.trState.reshape((1, -1)))
            self.trControl.append(phase.trControl.reshape((1, -1)))
            self.trSigma.append(phase.trSigma)

    def initCost(self, realJ, fakeJ):
        self.JTrace.append(realJ)
        self.fakeJTrace.append(fakeJ)
