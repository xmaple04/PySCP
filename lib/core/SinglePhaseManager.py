"""
author: xmaple
date: 2022-01-12
"""

import numpy as np
from lib.utils.xfuncs import RelativeError, PointwiseRelativeError, floatEqual
from scipy.integrate import odeint
from lib.Mesh.MeshSS import ZOH, FOH, RungeKutta, Trapezoidal
from lib.Mesh.MeshHP import hpLG, hpLGR, hpfLGR, hpPseudoSpectral
from lib.Mesh.MeshHybrid import Hybrid

meshMethodDict = {'ZOH': ZOH,
                  'FOH': FOH,
                  'RK': RungeKutta,
                  'Trapezoidal': Trapezoidal,

                  'LG': hpLG,
                  'LGR': hpLGR,
                  'fLGR': hpfLGR,
                  'Hybrid': Hybrid}

"""
Phase
"""


class PhaseManager:
    def __init__(self, model, phaseNo, meshConfig):
        """ ------------------------------- initialize mesh  ------------------------------- """
        assert meshConfig['meshName'] in meshMethodDict
        self.meshName = meshConfig['meshName']
        if self.isPseudospectral():
            self.scheme = meshConfig['scheme']
        else:
            self.scheme = None
        method = meshMethodDict[self.meshName]

        if self.isPseudospectral():
            # pseudo-spectral methods
            if isinstance(meshConfig['degree'], list):
                degree = meshConfig['degree']
            else:
                degree = [meshConfig['degree']]

            segFractions = meshConfig.get('segFractions')
            if segFractions is None:
                segFractions = [1. / len(degree)] * len(degree)
            else:
                assert floatEqual(sum(segFractions), 1.)

            self.mesh = method(degree=degree, segFractions=segFractions)
        elif self.isSingleStep():
            # single step methods
            self.mesh = method(ncp=meshConfig.get('ncp'), mps=meshConfig.get('mps'))
        elif self.meshName == 'Hybrid':
            segs = 4
            self.mesh = Hybrid(segDegrees=[10] * segs, segNames=['fLGR'] * segs, segFractions=[1 / segs] * segs)
            # self.mesh = Hybrid(segDegrees=[10, 10, 10, 10], segNames=['fLGR', 'RK', 'fLGR', 'RK'], segFractions=[0.25, 0.25, 0.25, 0.25])
        elif self.meshName == 'Trapezoidal':
            self.mesh = Trapezoidal(ncp=meshConfig.get('ncp'), mps=meshConfig.get('mps'))

        self.model = model
        self.phaseNo = phaseNo
        self.xdim, self.udim = model.phases[phaseNo].getDimension()
        self.ncp = self.mesh.ncp
        self.nx = self.mesh.nx
        self.nu = self.mesh.nu

        """ ------------------------------- approximate matrices  ------------------------------- """
        self.A_tilde = np.zeros([self.ncp, self.xdim, self.xdim])
        self.B_tilde = np.zeros([self.ncp, self.xdim, self.udim])
        self.B2_tilde = np.zeros([self.ncp, self.xdim, self.udim])
        self.f_tilde = np.zeros([self.ncp, self.xdim])
        self.R_tilde = np.zeros([self.ncp, self.xdim])

        if self.meshName == 'ZOH':
            # the state and all matrices must be stacked to a flat vector for integration
            # [x, A~, B~, f~, R~] the indices for the state and matrices are marked by index
            self.x_index = slice(0, self.xdim)
            self.A_tilde_index = slice(self.xdim, self.xdim * (1 + self.xdim))
            self.B_tilde_index = slice(self.xdim * (1 + self.xdim), self.xdim * (1 + self.xdim + self.udim))
            self.f_tilde_index = slice(self.xdim * (1 + self.xdim + 2 * self.udim), self.xdim * (2 + self.xdim + 2 * self.udim))
            self.R_tilde_index = slice(self.xdim * (2 + self.xdim + 2 * self.udim), self.xdim * (3 + self.xdim + 2 * self.udim))
            self.vec0 = np.zeros((self.xdim * (3 + self.xdim + 2 * self.udim),))
            self.vec0[self.A_tilde_index] = np.eye(self.xdim).flatten()  # only x is changeable.
        elif self.meshName == 'FOH':
            self.x_index = slice(0, self.xdim)
            self.A_tilde_index = slice(self.xdim, self.xdim * (1 + self.xdim))
            self.B_tilde_index = slice(self.xdim * (1 + self.xdim), self.xdim * (1 + self.xdim + self.udim))
            self.B2_tilde_index = slice(self.xdim * (1 + self.xdim + self.udim), self.xdim * (1 + self.xdim + 2 * self.udim))
            self.f_tilde_index = slice(self.xdim * (1 + self.xdim + 2 * self.udim), self.xdim * (2 + self.xdim + 2 * self.udim))
            self.R_tilde_index = slice(self.xdim * (2 + self.xdim + 2 * self.udim), self.xdim * (3 + self.xdim + 2 * self.udim))
            self.vec0 = np.zeros((self.xdim * (3 + self.xdim + 2 * self.udim),))
            self.vec0[self.A_tilde_index] = np.eye(self.xdim).flatten()  # only x is changeable.
        elif self.meshName == 'RK':
            self.x_index = slice(0, self.xdim)
            self.A_tilde_index = slice(self.xdim, self.xdim * (1 + self.xdim))
            self.B_tilde_index = slice(self.xdim * (1 + self.xdim), self.xdim * (1 + self.xdim + self.udim))
            self.B2_tilde_index = slice(self.xdim * (1 + self.xdim + self.udim), self.xdim * (1 + self.xdim + 2 * self.udim))
            self.f_tilde_index = slice(self.xdim * (1 + self.xdim + 2 * self.udim), self.xdim * (2 + self.xdim + 2 * self.udim))
            self.R_tilde_index = slice(self.xdim * (2 + self.xdim + 2 * self.udim), self.xdim * (3 + self.xdim + 2 * self.udim))
        elif self.meshName == 'Trapezoidal':
            self.f_tilde = np.zeros([self.ncp + 1, self.xdim])
            self.A_tilde = np.zeros([self.ncp + 1, self.xdim, self.xdim])
            self.B_tilde = np.zeros([self.ncp + 1, self.xdim, self.udim])

    def getDynamicsFuncValue(self, x, u, t, auxdata):
        # 定常系统的动力学方程只与状态和控制有关，与时间无关。为了以后适用于时变系统，留了时间变量接口
        return self.model.phases[self.phaseNo].dynamicsFunc(x, u, 0, auxdata)

    def refreshMatrices(self, xk, uk, sigmak, auxdata):
        if self.isPseudospectral():
            for i in range(self.ncp):
                ix = self.mesh.cpState[i]
                ff, A, B = self.getDynamicsFuncValue(xk[ix], uk[i], 0, auxdata)
                self.A_tilde[i] = sigmak * A
                self.B_tilde[i] = sigmak * B
                self.f_tilde[i] = ff
                self.R_tilde[i] = -sigmak * (np.matmul(A, xk[ix]) + np.matmul(B, uk[i]))
        elif self.meshName == 'ZOH':
            for i in range(self.ncp):
                dt = self.mesh.xtao[i + 1] - self.mesh.xtao[i]  # interval length
                self.vec0[self.x_index] = xk[i]
                vecf = np.array(odeint(self.ZOHDynamics, self.vec0, (0, dt), args=(uk[i], sigmak, auxdata,))[1])

                phi = vecf[self.A_tilde_index].reshape((self.xdim, self.xdim))
                self.A_tilde[i] = phi
                self.B_tilde[i] = np.matmul(phi, vecf[self.B_tilde_index].reshape((self.xdim, self.udim)))
                self.f_tilde[i] = np.matmul(phi, vecf[self.f_tilde_index])
                self.R_tilde[i] = np.matmul(phi, vecf[self.R_tilde_index])
        elif self.meshName == 'FOH':
            for i in range(self.ncp):
                dt = self.mesh.xtao[i + 1] - self.mesh.xtao[i]  # interval length
                self.vec0[self.x_index] = xk[i]
                vecf = np.array(odeint(self.FOHDynamics, self.vec0, (0, dt), args=(uk[i], uk[i + 1], dt, sigmak, auxdata))[1])

                phi = vecf[self.A_tilde_index].reshape((self.xdim, self.xdim))
                self.A_tilde[i] = phi
                self.B_tilde[i] = np.matmul(phi, vecf[self.B_tilde_index].reshape((self.xdim, self.udim)))
                self.B2_tilde[i] = np.matmul(phi, vecf[self.B2_tilde_index].reshape((self.xdim, self.udim)))
                self.f_tilde[i] = np.matmul(phi, vecf[self.f_tilde_index])
                self.R_tilde[i] = np.matmul(phi, vecf[self.R_tilde_index])
        elif self.meshName == 'RK':
            for i in range(self.ncp):
                um = (uk[i] + uk[i + 1]) / 2  # U_(i+1/2)
                dt = self.mesh.xtao[i + 1] - self.mesh.xtao[i]

                ff1, A1, B1 = self.getDynamicsFuncValue(xk[i], uk[i], 0, auxdata)
                sA1 = sigmak * A1
                sB1 = sigmak * B1
                R1 = -(np.matmul(sA1, xk[i]) + np.matmul(sB1, uk[i]))
                k1 = sigmak * ff1

                xk_m1 = xk[i] + dt / 2 * k1
                ff2, A2, B2 = self.getDynamicsFuncValue(xk_m1, um, 0, auxdata)
                sA2 = sigmak * A2
                sB2 = sigmak * B2
                R2 = -(np.matmul(sA2, xk_m1) + np.matmul(sB2, um))
                k2 = sigmak * ff2

                xk_m2 = xk[i] + dt / 2 * k2
                ff3, A3, B3 = self.getDynamicsFuncValue(xk_m2, um, 0, auxdata)
                sA3 = sigmak * A3
                sB3 = sigmak * B3
                R3 = -(np.matmul(sA3, xk_m2) + np.matmul(sB3, um))
                k3 = sigmak * ff3

                xk1 = xk[i] + dt * k3
                ff4, A4, B4 = self.getDynamicsFuncValue(xk1, uk[i + 1], 0, auxdata)
                sA4 = sigmak * A4
                sB4 = sigmak * B4
                R4 = -(np.matmul(sA4, xk1) + np.matmul(sB4, uk[i + 1]))
                k4 = sigmak * ff4

                unitMat = np.eye(self.xdim)
                coef1 = dt ** 3 / 4 * np.matmul(sA4, np.matmul(sA3, sA2)) + dt ** 2 / 2 * np.matmul(sA3, sA2) + dt * sA2 + unitMat
                coef2 = dt ** 2 / 4 * np.matmul(sA4, sA3) + dt / 2 * sA3 + unitMat
                coef3 = dt / 2 * sA4 + unitMat
                coef4 = unitMat

                self.A_tilde[i] = (unitMat + dt / 6 * (np.matmul(coef1, sA1) + 2 * np.matmul(coef2, sA2) + 2 * np.matmul(coef3, sA3) + np.matmul(coef4, sA4)))
                self.B_tilde[i] = (dt / 6 * (np.matmul(coef1, sB1) + np.matmul(coef2, sB2) + np.matmul(coef3, sB3)))
                self.B2_tilde[i] = (dt / 6 * (np.matmul(coef2, sB2) + np.matmul(coef3, sB3) + np.matmul(coef4, sB4)))
                self.f_tilde[i] = dt / 6 * (np.matmul(coef1, ff1) + 2 * np.matmul(coef2, ff2) + 2 * np.matmul(coef3, ff3) + np.matmul(coef4, ff4))
                self.R_tilde[i] = dt / 6 * (np.matmul(coef1, R1) + 2 * np.matmul(coef2, R2) + 2 * np.matmul(coef3, R3) + np.matmul(coef4, R4))
        elif self.meshName == 'Hybrid':
            imat = 0
            for iseg in range(self.mesh.nseg):
                segXk = xk[self.mesh.segStateIndex[iseg]]
                segUk = uk[self.mesh.segControlIndex[iseg]]
                if self.mesh.segNames[iseg] == 'RK':
                    for i in range(self.mesh.segDegrees[iseg]):
                        um = (segUk[i] + segUk[i + 1]) / 2  # U_(i+1/2)
                        dt = self.mesh.xtao[imat + 1] - self.mesh.xtao[imat]

                        ff1, A1, B1 = self.getDynamicsFuncValue(segXk[i], segUk[i], 0, auxdata)
                        sA1 = sigmak * A1
                        sB1 = sigmak * B1
                        R1 = -(np.matmul(sA1, segXk[i]) + np.matmul(sB1, segUk[i]))
                        k1 = sigmak * ff1

                        xk_m1 = segXk[i] + dt / 2 * k1
                        ff2, A2, B2 = self.getDynamicsFuncValue(xk_m1, um, 0, auxdata)
                        sA2 = sigmak * A2
                        sB2 = sigmak * B2
                        R2 = -(np.matmul(sA2, xk_m1) + np.matmul(sB2, um))
                        k2 = sigmak * ff2

                        xk_m2 = segXk[i] + dt / 2 * k2
                        ff3, A3, B3 = self.getDynamicsFuncValue(xk_m2, um, 0, auxdata)
                        sA3 = sigmak * A3
                        sB3 = sigmak * B3
                        R3 = -(np.matmul(sA3, xk_m2) + np.matmul(sB3, um))
                        k3 = sigmak * ff3

                        xk1 = segXk[i] + dt * k3
                        ff4, A4, B4 = self.getDynamicsFuncValue(xk1, segUk[i + 1], 0, auxdata)
                        sA4 = sigmak * A4
                        sB4 = sigmak * B4
                        R4 = -(np.matmul(sA4, xk1) + np.matmul(sB4, segUk[i + 1]))
                        k4 = sigmak * ff4

                        unitMat = np.eye(self.xdim)
                        coef1 = dt ** 3 / 4 * np.matmul(sA4, np.matmul(sA3, sA2)) + dt ** 2 / 2 * np.matmul(sA3, sA2) + dt * sA2 + unitMat
                        coef2 = dt ** 2 / 4 * np.matmul(sA4, sA3) + dt / 2 * sA3 + unitMat
                        coef3 = dt / 2 * sA4 + unitMat
                        coef4 = unitMat

                        self.A_tilde[imat] = (unitMat + dt / 6 * (np.matmul(coef1, sA1) + 2 * np.matmul(coef2, sA2) + 2 * np.matmul(coef3, sA3) + np.matmul(coef4, sA4)))
                        self.B_tilde[imat] = (dt / 6 * (np.matmul(coef1, sB1) + np.matmul(coef2, sB2) + np.matmul(coef3, sB3)))
                        self.B2_tilde[imat] = (dt / 6 * (np.matmul(coef2, sB2) + np.matmul(coef3, sB3) + np.matmul(coef4, sB4)))
                        self.f_tilde[imat] = dt / 6 * (np.matmul(coef1, ff1) + 2 * np.matmul(coef2, ff2) + 2 * np.matmul(coef3, ff3) + np.matmul(coef4, ff4))
                        self.R_tilde[imat] = dt / 6 * (np.matmul(coef1, R1) + 2 * np.matmul(coef2, R2) + 2 * np.matmul(coef3, R3) + np.matmul(coef4, R4))
                        imat += 1
                else:
                    for i in range(self.mesh.segDegrees[iseg]):
                        ff, A, B = self.getDynamicsFuncValue(segXk[i + 1], segUk[i], 0, auxdata)
                        self.A_tilde[imat] = sigmak * A
                        self.B_tilde[imat] = sigmak * B
                        self.f_tilde[imat] = ff
                        self.R_tilde[imat] = -sigmak * (np.matmul(A, segXk[i + 1]) + np.matmul(B, segUk[i]))
                        imat += 1
        elif self.meshName == 'Trapezoidal':
            for i in range(self.ncp + 1):
                ff, A, B = self.getDynamicsFuncValue(xk[i], uk[i], 0, auxdata)
                self.f_tilde[i] = ff
                self.A_tilde[i] = A
                self.B_tilde[i] = B

    def ZOHDynamics(self, vec, t, u, sigma, auxdata):
        """
        derivative function to compute state matrix.

        :param vec: Evaluation state V = [x, A~, B~, f~, R~]
        :param t: Evaluation time
        :param u: control at start of interval
        :param sigma: time scale
        :param dt: interval length
        :return: Derivative at current time
        """
        x = vec[self.x_index]

        ff, A, B = self.getDynamicsFuncValue(x, u, t, auxdata)

        Asigma = A.reshape((self.xdim, self.xdim)) * sigma
        Bsigma = B.reshape((self.xdim, self.udim)) * sigma
        dvdt = np.zeros_like(vec)

        A_tilde = vec[self.A_tilde_index].reshape((self.xdim, self.xdim))
        phi_inv = np.linalg.inv(A_tilde)
        dvdt[self.x_index] = sigma * ff.transpose()
        dvdt[self.A_tilde_index] = np.matmul(Asigma, A_tilde).flatten()
        dvdt[self.B_tilde_index] = np.matmul(phi_inv, Bsigma).flatten()
        dvdt[self.f_tilde_index] = np.matmul(phi_inv, ff).transpose()
        RR = -np.matmul(Asigma, x) - np.matmul(Bsigma, u)
        dvdt[self.R_tilde_index] = np.matmul(phi_inv, RR)

        return dvdt

    def FOHDynamics(self, vec, t, uk, uk_1, dt, sigma, auxdata):
        """
        derivative function to compute state matrix.

        :param vec: Evaluation state V = [x, A~, B~, f~, R~]
        :param t: Evaluation time
        :param u: control at start of interval
        :param sigma: time scale
        :param dt: interval length
        :return: Derivative at current time
        """
        alpha = 1 - t / dt
        beta = t / dt
        u = alpha * uk + beta * uk_1
        x = vec[self.x_index]

        ff, A, B = self.getDynamicsFuncValue(x, u, t, auxdata)

        Asigma = A.reshape((self.xdim, self.xdim)) * sigma
        Bsigma = B.reshape((self.xdim, self.udim)) * sigma
        dvdt = np.zeros_like(vec)

        A_tilde = vec[self.A_tilde_index].reshape((self.xdim, self.xdim))
        phi_inv = np.linalg.inv(A_tilde)
        dvdt[self.x_index] = sigma * ff.transpose()
        dvdt[self.A_tilde_index] = np.matmul(Asigma, A_tilde).flatten()
        dvdt[self.B_tilde_index] = np.matmul(phi_inv, Bsigma).flatten() * alpha
        dvdt[self.B2_tilde_index] = np.matmul(phi_inv, Bsigma).flatten() * beta
        dvdt[self.f_tilde_index] = np.matmul(phi_inv, ff).transpose()
        RR = -(np.matmul(Asigma, x) + np.matmul(Bsigma, u))
        dvdt[self.R_tilde_index] = np.matmul(phi_inv, RR)

        return dvdt

    def getHPFunctionValue(self, xk, uk, sigmak, auxdata):
        """ get integrated states """
        fx = np.zeros((self.ncp, self.xdim))
        for i in range(self.ncp):
            ix = self.mesh.cpState[i]
            fx[i], _, _ = self.getDynamicsFuncValue(xk[ix], uk[i], sigmak, auxdata)
        return fx

    def getHybridFunctionValue(self, xk, uk, sigmak, auxdata):
        """ get integrated states """
        fx = np.zeros((len(uk), self.xdim))
        for i in range(len(uk)):
            fx[i], _, _ = self.getDynamicsFuncValue(xk[i + 1], uk[i], sigmak, auxdata)
        return fx

    def getApproximateMatrices(self):
        """ get approximation matrices: A_tilde, B_tilde, B2_tilde, F_tilde, R_tilde """
        return self.A_tilde, self.B_tilde, self.B2_tilde, self.f_tilde, self.R_tilde

    def isPseudospectral(self):
        if self.meshName in ['LG', 'LGR', 'fLGR']:
            return True
        else:
            return False

    def isSingleStep(self):
        if self.meshName in ['ZOH', 'FOH', 'RK']:
            return True
        else:
            return False

    def isHybrid(self):
        if self.meshName in ['Hybrid']:
            return True
        else:
            return False

    def isTrapezoidal(self):
        if self.meshName in ['Trapezoidal']:
            return True
        else:
            return False

    def getPointwiseDynamicsCost(self, xk, uk, sigmak, auxdata):
        """ 动力学方程误差 DX-f(x,u) """
        if self.isPseudospectral():
            fx = self.getHPFunctionValue(xk, uk, sigmak, auxdata)
            if self.scheme == 'differential':
                cost = PointwiseRelativeError(np.matmul(self.mesh.PDM, xk), sigmak * fx)
                return cost
            else:
                cost = PointwiseRelativeError(xk[self.mesh.PIMXfIndex], xk[self.mesh.PIMX0Index] + sigmak * np.matmul(self.mesh.PIM, fx))
                return cost
        elif self.isSingleStep():
            Xint = np.zeros((self.ncp, self.xdim))
            if self.meshName == 'ZOH':
                for i in range(self.ncp):
                    Xint[i] = (self.A_tilde[i] @ xk[i]
                               + self.B_tilde[i] @ uk[i]
                               + self.f_tilde[i] * sigmak
                               + self.R_tilde[i])
            else:
                for i in range(self.ncp):
                    Xint[i] = (self.A_tilde[i] @ xk[i]
                               + self.B_tilde[i] @ uk[i] + self.B2_tilde[i] @ uk[i + 1]
                               + self.f_tilde[i] * sigmak
                               + self.R_tilde[i])
            return PointwiseRelativeError(xk[1:], Xint)
