"""
author: xmaple
date: 2022-03-22
"""


def RungeKutta(x, t):
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
