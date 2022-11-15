import matplotlib.pyplot as plt
import numpy as np
import cyipopt

from scipy.integrate import odeint
from scipy.interpolate import interp1d
from lib.utils.plotlib import *
from lib.utils.PlanetData import Earth
from lib.utils.Scale import Scale
from lib.Mesh.MeshHP import hpfLGR


class Vehicle:
    def __init__(self, ncp):
        self.ncp = ncp
        self.xdim = 2
        self.udim = 1
        self.lgr = hpfLGR([ncp], [1])
        self.PDM = self.lgr.PDM
        self.state0 = np.array([0, 0])
        self.statef = np.array([4.5, 3])

        self.planet = Earth()
        self.scale = Scale(length=100e3, velocity=3000,
                           state_name=['length', 'velocity', 'angle', 'non'],
                           control_name=['rate'])

    def objective(self, x):
        state, control, tf = self.x2sct(x)
        dxDiff = np.matmul(self.lgr.PDM, state) - tf / 2 * self.fx(state, control)
        # lineCost = np.linalg.norm(dxDiff, axis=1, ord=2)
        # cost = np.linalg.norm(lineCost, ord=np.inf)
        cost = np.sqrt(np.sum(dxDiff ** 2))
        return cost

    def fx(self, state, control):
        f = np.zeros((self.ncp, 2))
        for i in range(self.ncp):
            f[i, 0] = state[i + 1, 1]
            f[i, 1] = control[i, 0]
        return f

    def gradient(self, x):
        state, control, tf = self.x2sct(x)
        dxDiff = np.matmul(self.lgr.PDM, state) - tf / 2 * self.fx(state, control)
        normDer = self.vector2Dgrad(dxDiff)

        partPDM = self.PDM[:, 1:]
        xgrad = np.zeros((self.ncp, self.xdim))
        ugrad = np.zeros((self.ncp, self.udim))
        f, A, B = self.Dynamics(state, control)
        for m in range(self.ncp):
            # xgrad
            for n in range(self.xdim):
                for i in range(self.ncp):
                    for j in range(self.xdim):
                        if j == n:
                            xgrad[m, n] += normDer[i, j] * partPDM[i, m]
                        if i == m:
                            xgrad[m, n] += - normDer[i, j] * tf / 2 * A[i, j, n]
            # ugrad
            for n in range(self.udim):
                for i in range(self.ncp):
                    for j in range(self.xdim):
                        if i == m:
                            ugrad[m, n] += - normDer[i, j] * tf / 2 * B[i, j, n]

        tgrad = -1 / 2 * np.sum(f)
        grad = np.hstack([xgrad.flatten(order='F'), ugrad.flatten(order='F'), tgrad])
        return grad

    def vector2Dgrad(self, cost2D):
        # lineCost = np.linalg.norm(cost2D, axis=1, ord=2)
        # flags = self.infinityNorm(lineCost)
        # xgrad = cost2D / lineCost.reshape((-1, 1)) * flags
        xgrad = cost2D / np.sqrt(np.sum(cost2D ** 2))
        return xgrad

    def infinityNorm(self, x):
        cost = np.linalg.norm(x, ord=np.inf)
        flags = np.zeros_like(x)
        flags[np.where(np.abs(x) == cost)] = 1
        num = np.sum(np.abs(flags))
        flags = flags / num
        return flags.reshape((-1, 1))

    def Dynamics(self, state, control):
        f = np.zeros((self.ncp, self.xdim))
        A = np.zeros((self.ncp, self.xdim, self.xdim))
        B = np.zeros((self.ncp, self.xdim, self.udim))
        for i in range(self.ncp):
            x = state[i + 1]
            u = control[i]
            f[i, 0] = x[1]
            f[i, 1] = u[0]
            A[i] = np.array([[0, 1], [0, 0]])
            B[i] = np.array([[0], [1]])
        return f, A, B

    # def constraints(self, x):
    #     pass
    #
    # def jacobian(self, x):
    #     pass

    def getInitVector(self, xlb, xub, xflb, xfub, ulb, uub, tflb, tfub):
        lb = np.repeat(xlb, self.ncp, axis=0)
        for i in range(len(xflb)):
            lb[(i + 1) * self.ncp - 1] = xlb[i]
        lb = np.concatenate([lb, np.repeat(ulb, self.ncp), [tflb]])

        ub = np.repeat(xub, self.ncp, axis=0)
        for i in range(len(xfub)):
            ub[(i + 1) * self.ncp - 1] = xub[i]
        ub = np.concatenate([ub, np.repeat(uub, self.ncp), [tfub]])

        wx = (1 + self.lgr.xtao) / 2
        wx = wx.reshape((-1, 1))
        state = (1 - wx) * self.state0.reshape((1, -1)) + wx * self.statef.reshape((1, -1))
        state = state[1:].flatten(order='F')

        wu = (1 + self.lgr.utao) / 2
        wu = wu.reshape((-1, 1))
        control = (1 - wu) * 1 + wu * 1
        control = control.flatten(order='F')
        x0 = np.concatenate([state, control, [(tflb + tfub) / 2]])
        # self.plot(x0)
        # self.plot(lb)
        # self.plot(ub)
        # x = self.objective(x0)
        return x0, lb, ub

    def x2sct(self, x):
        hIndex = slice(0, self.ncp)
        vIndex = slice(self.ncp, 2 * self.ncp)
        uIndex = slice(2 * self.ncp, -1)
        h = np.concatenate([[self.state0[0]], x[hIndex]]).reshape((-1, 1))
        v = np.concatenate([[self.state0[1]], x[vIndex]]).reshape((-1, 1))
        u = x[uIndex].reshape((-1, 1))
        tf = x[-1]
        state = np.hstack([h, v])
        control = u
        return state, control, tf

    def plot(self, x):
        state, control, tf = self.x2sct(x)
        stateInt = self.propogate(x)
        tint = np.linspace(0, tf, len(stateInt))
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot((self.lgr.xtao + 1) * tf / 2, state[:, 0])
        plt.plot(tint, stateInt[:, 0])
        plt.subplot(2, 2, 2)
        plt.plot((self.lgr.xtao + 1) * tf / 2, state[:, 1])
        plt.plot(tint, stateInt[:, 1])
        plt.subplot(2, 2, 3)
        plt.plot((self.lgr.utao + 1) * tf / 2, control[:, 0])
        plt.show()

    def func(self, x, t, control, tf):
        ufun = interp1d(self.lgr.utao, control.flatten(), fill_value='extrapolate')
        return np.array([x[1], ufun(2 * t / tf - 1)])

    def propogate(self, x):
        state, control, tf = self.x2sct(x)
        x0 = self.state0.copy()
        t = np.linspace(0, tf, 100)
        history = odeint(self.func, x0, t, args=(control, tf,))
        return history

    # def Dynamics(self, x, u, scale):
    #     Bc = 1e3
    #     LD = 3
    #     h, v, gamma = x
    #     lam = u[0]
    #
    #     Re_scaled = self.planet.radius / scale.length
    #     mu_scaled = self.planet.mu / (scale.length ** 3 / scale.time ** 2)
    #     H0_scaled = self.planet.H0 / scale.length
    #
    #     rho = self.planet.rho0 * np.exp(-h / H0_scaled) * scale.length
    #     q = 1 / 2 * rho * v ** 2
    #
    #     r = h + Re_scaled
    #     f = np.zeros_like(x)
    #     f[0] = v * np.sin(gamma)
    #     f[1] = - mu_scaled / (r ** 2) * np.sin(gamma) - (1 + lam ** 2) / 2 * q / Bc
    #     f[2] = (v / r - mu_scaled / ((r ** 2) * v)) * np.cos(gamma) + lam * LD / v * q / Bc
    #     return f


if __name__ == '__main__':
    from scipy.optimize import minimize

    vehicle = Vehicle(20)
    xlb = np.array([0, 0])
    xub = np.array([9, 5])
    xflb = np.array([4, 2.9])
    xfub = np.array([5, 3.1])
    ulb = 0
    uub = 2
    tflb = 2
    tfub = 6

    x0, lb, ub = vehicle.getInitVector(xlb, xub, xflb, xfub, ulb, uub, tflb, tfub)
    bounds = [(lb[i], ub[i]) for i in range(len(lb))]
    ret = minimize(vehicle.objective, x0, method='SLSQP', jac=vehicle.gradient, bounds=bounds, ftol=1e-6,
                   options={'maxiter': 1000})
    x = ret.x
    print('目标函数值：', ret.fun, vehicle.objective(x))
    print('是否成功', ret.success)
    vehicle.plot(ret.x)
