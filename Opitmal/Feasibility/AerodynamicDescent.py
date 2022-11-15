import matplotlib.pyplot as plt
import numpy as np
import cyipopt
import time

from scipy.integrate import odeint
from scipy.interpolate import interp1d
from lib.utils.plotlib import *
from lib.utils.PlanetData import Earth
from lib.utils.Scale import Scale
from lib.Mesh.MeshHP import hpfLGR


def quadraticInterp(t0, y0, t1):
    """ 二次插值 """
    f = interp1d(t0, y0, kind='quadratic', fill_value='extrapolate')
    y1 = f(t1)
    return y1


class Vehicle:
    def __init__(self, ncp, x0lb, x0ub, xlb, xub, xflb, xfub, ulb, uub, tflb, tfub):
        self.x0lb = x0lb
        self.x0ub = x0ub
        self.xlb = xlb
        self.xub = xub
        self.xflb = xflb
        self.xfub = xfub
        self.ulb = ulb
        self.uub = uub
        self.tflb = tflb
        self.tfub = tfub

        self.ncp = ncp
        self.nx = ncp + 1
        self.xdim = 2
        self.udim = 1
        self.xslices = self.getSlices()

        self.lgr = hpfLGR([ncp], [1])
        self.PDM = self.lgr.PDM

        self.x0, self.lb, self.ub = self.initialize()

        self.planet = Earth()
        self.scale = Scale(length=100e3, velocity=3000,
                           state_name=['length', 'velocity', 'angle', 'non'],
                           control_name=['rate'])
        self.tt = 0

    def getSlices(self):
        assert np.all(self.x0lb >= self.xlb)
        assert np.all(self.xflb >= self.xlb)
        assert np.all(self.x0ub <= self.xub)
        assert np.all(self.xfub <= self.xub)
        slices = []
        for i in range(self.xdim):
            a = 0
            b = self.ncp + 1
            if self.x0lb[i] == self.x0ub[i]:
                a = 1
            if self.xflb[i] == self.xfub[i]:
                b = b - 1
            slices.append(slice(a, b))
        return slices

    def initialize(self):
        lb = []
        ub = []
        for i in range(self.xdim):
            if self.xslices[i].start == 0:  # 初始状态自由
                lb.append(self.x0lb[i])
                ub.append(self.x0ub[i])
            lb.append([self.xlb[i]] * (self.ncp - 1))
            ub.append([self.xub[i]] * (self.ncp - 1))
            if self.xslices[i].stop == self.ncp + 1:  # 终端状态自由
                lb.append(self.xflb[i])
                ub.append(self.xfub[i])
        for i in range(self.udim):
            lb.append([self.ulb[i]] * self.ncp)
            ub.append([self.uub[i]] * self.ncp)
        lb.append(tflb)
        ub.append(tfub)
        lb = np.hstack(lb)
        ub = np.hstack(ub)

        wx = (1 + self.lgr.xtao) / 2
        wx = wx.reshape((-1, 1))
        state = (1 - wx) * (self.x0lb + self.x0ub) / 2 + wx * (self.xflb + self.xfub) / 2
        states = []
        for i in range(self.xdim):
            states.append(state[self.xslices[i], i])
        state = np.hstack(states)

        wu = (1 + self.lgr.utao) / 2
        wu = wu.reshape((-1, 1))
        control = (1 - wu) * 1 + wu * 1
        control = control.flatten(order='F')
        control = np.ones_like(control) * 2
        x0 = np.concatenate([state, control, [(self.tflb + self.tfub) / 2]])
        return x0, lb, ub

    def refine(self, mesh, x, u):
        xtao = mesh.xtao
        utao = mesh.utao

        def plot_trial_state(nodes, state_k, trial_nodes, trial_state):
            import matplotlib.pyplot as plt
            plt.figure()
            plt.switch_backend('QT5Agg')
            axes = []
            for i in range(state_k.shape[0]):
                axes.append(plt.subplot(2, 3, i + 1))
                axes[i].plot(nodes, state_k[i, :], marker='o', markerfacecolor='none')
                axes[i].plot(trial_nodes, trial_state[i, :], marker='s', markerfacecolor='none')

            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()

        def get_seg_error(mesh_k, X_k, U_k, trial_mesh, trial_state, trial_U, sigma_k):
            hp_1 = mesh_k.hp
            segments = len(hp_1)
            nv_origin = self.get_dif_nv(mesh_k, X_k, U_k, sigma_k)  # 原mesh的误差
            nv_trial = self.get_dif_nv(trial_mesh, trial_state, trial_U, sigma_k)  # 新mesh的误差
            seg_error_origin = np.zeros((segments,))
            seg_error_trial = np.zeros((segments,))
            index_origin = mesh_k.segwise_collop_from_collop()
            index_trial = trial_mesh.segwise_collop_from_collop()
            for i, p in enumerate(hp_1):
                seg_error_origin[i] = np.linalg.norm(nv_origin[:, index_origin[i]])
                seg_error_trial[i] = np.linalg.norm(nv_trial[:, index_trial[i]])
            return seg_error_origin, seg_error_trial

        def plot_trial_error(mesh_k, X_k, U_k, trial_mesh, trial_state, trial_U, sigma_k):
            import matplotlib.pyplot as plt
            plt.figure()
            plt.switch_backend('QT5Agg')
            axes = []
            nu_k = self.get_dif_nv(mesh_k, X_k, U_k, sigma_k)
            trial_k = self.get_dif_nv(trial_mesh, trial_state, trial_U, sigma_k)
            # nu_k = self.get_int_nv(mesh_k, X_k, U_k, sigma_k)
            # trial_k = self.get_int_nv(trial_mesh, trial_state, trial_U, sigma_k)
            for i in range(X_k.shape[0]):
                axes.append(plt.subplot(2, 3, i + 1))
                axes[i].plot(mesh_k.nodes[1:], nu_k[i, :], marker='o', markerfacecolor='none')
                axes[i].plot(trial_mesh.nodes[1:], trial_k[i, :], marker='s', markerfacecolor='none')
                axes[i].legend(['current cost', 'approximate cost'])

            # 配点误差绝对值之和
            plt.subplot(2, 3, 6)
            nu = np.linalg.norm(nu_k, axis=0)
            plt.plot(mesh_k.nodes[1:], nu, marker='o', markerfacecolor='none')
            trial = np.linalg.norm(trial_k, axis=0)
            plt.plot(trial_mesh.nodes[1:], trial, marker='s', markerfacecolor='none')
            plt.yscale('log')
            plt.legend(['current cost', 'approximate cost'])

            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()

        def plotProcess(trial_error, origin_error):
            import matplotlib.pyplot as plt
            plt.figure()
            num = len(trial_error)
            ratios = []
            for i in range(num):
                ratio = (trial_error[i] / origin_error[i])
                ratios.append(ratio)
            ratios = np.array(ratios)
            width = 0.2
            for i in range(ratios.shape[1]):
                plt.bar(np.arange(num) + i * width, ratios[:, i], width / 2, color='salmon', label='类别A')
            plt.show()

    def objective(self, x):
        state, control, tf = self.x2sct(x)
        dxDiff = np.matmul(self.lgr.PDM, state) - tf / 2 * self.fx(state, control)
        linecost = np.sqrt(np.sum(dxDiff ** 2, axis=1))
        cost = np.matmul(self.lgr.weights4Control, linecost)
        return cost

    def fx(self, state, control):
        f = np.zeros((self.ncp, 2))
        for i in range(self.ncp):
            f[i, 0] = state[i + 1, 1]
            f[i, 1] = control[i, 0]
        return f

    def gradient(self, x):
        start = time.time()
        state, control, tf = self.x2sct(x)
        dxDiff = np.matmul(self.lgr.PDM, state) - tf / 2 * self.fx(state, control)
        linecost = np.sqrt(np.sum(dxDiff ** 2, axis=1))
        normDer = self.lgr.weights4Control.reshape((-1, 1)) * dxDiff / linecost.reshape((-1, 1))

        xgrad = np.zeros((self.nx, self.xdim))
        ugrad = np.zeros((self.ncp, self.udim))
        f, A, B = self.Dynamics(state, control)
        # xgrad
        for m in range(self.nx):
            for n in range(self.xdim):
                for i in range(self.ncp):
                    for j in range(self.xdim):
                        if j == n:
                            xgrad[m, n] += normDer[i, j] * self.PDM[i, m]
                        if i == m:
                            xgrad[m, n] += - normDer[i, j] * tf / 2 * A[i, j, n]
        # ugrad
        for m in range(self.ncp):
            for n in range(self.udim):
                for i in range(self.ncp):
                    for j in range(self.xdim):
                        if i == m:
                            ugrad[m, n] += - normDer[i, j] * tf / 2 * B[i, j, n]
        xgrads = []
        for i in range(self.xdim):
            xgrads.append(xgrad[self.xslices[i], i])
        xgrad = np.hstack(xgrads)

        tgrad = -1 / 2 * np.sum(normDer * f)
        grad = np.hstack([xgrad, ugrad.flatten(order='F'), tgrad])
        end = time.time()
        self.tt += end - start
        return grad

    def vector2Dgrad(self, cost2D):
        linecost = np.sqrt(np.sum(cost2D ** 2, axis=1))
        xgrad = self.lgr.weights4Control.reshape((-1, 1)) * cost2D / linecost.reshape((-1, 1))
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

    def x2sct(self, x):
        start = 0
        stop = 0
        state = np.zeros((self.nx, self.xdim))
        for i in range(self.xdim):
            state[0, i] = self.x0lb[i]
            state[-1, i] = self.xflb[i]
            num = self.xslices[i].stop - self.xslices[i].start
            stop += num
            state[self.xslices[i], i] = x[start:stop]
            start = stop

        control = np.zeros((self.ncp, self.udim))
        for i in range(self.udim):
            stop += self.ncp
            control[:, i] = x[start:stop]
            start = stop
        tf = x[-1]
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
        x0 = state[0]
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
    xlb = np.array([0., 0.])
    xub = np.array([6., 5.])
    x0lb = np.array([0., 0.])
    x0ub = np.array([0., 0.])
    xflb = np.array([4., 4.])
    xfub = np.array([4., 4.])
    ulb = [0.]
    uub = [2.0]
    tflb = 1.
    tfub = 2.
    vehicle = Vehicle(10, x0lb, x0ub, xlb, xub, xflb, xfub, ulb, uub, tflb, tfub)

    nlp = cyipopt.Problem(
        n=len(vehicle.x0),
        m=0,
        problem_obj=vehicle,
        lb=vehicle.lb,
        ub=vehicle.ub,
        # cl=cl,
        # cu=cu
    )

    # Set solver options
    # nlp.addOption('derivative_test', 'second-order')
    # nlp.add_option('mu_strategy', 'adaptive')
    nlp.addOption('max_iter', 1000)
    nlp.addOption('tol', 1e-3)
    nlp.addOption('dual_inf_tol', 1e2)
    nlp.add_option('acceptable_obj_change_tol', 1e-8)
    nlp.addOption('compl_inf_tol', 1e-1)
    # Scale the problem (Just for demonstration purposes)
    # nlp.set_problem_scaling(obj_scaling=1, x_scaling=[1] * len(lb))
    nlp.add_option('nlp_scaling_method', 'user-scaling')

    # Solve the problem
    x, info = nlp.solve(vehicle.x0)
    print("Solution of the primal variables: x=%s\n" % repr(x))
    print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))
    print("Objective=%s\n" % repr(info['obj_val']))

    vehicle.plot(x)
    vehicle.objective(x)
