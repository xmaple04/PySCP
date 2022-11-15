"""
author: xmaple
date: 2022-01-01
TODO:
"""
import numpy as np
from lib.utils.PlanetData import Earth
from lib.utils.Scale import Scale
from scipy.integrate import odeint
from pyswarms.single.global_best import GlobalBestPSO
from lib.utils.plotlib import *


def CostateDynamics(x, u, params):
    h, d, v, gm, z = x
    ita, delta, sig = u
    Bc = params['Bc']  # 弹道系数
    LD = params['LD']  # 最大升阻比
    tw = params['tw']  # 初始推重比
    Isp = params['Isp'] / params['scale'].time  # 比冲

    Re = Earth.radius / params['scale'].length  # 地球半径
    mu = Earth.mu / (params['scale'].length ** 3 / params['scale'].time ** 2)  # 地球引力常数
    beta = 1 / (Earth.H0 / params['scale'].length)  # 参考高度的倒数
    q = 1 / 2 * Earth.rho0 * np.exp(-h * beta) * v ** 2 * params['scale'].length  # 动压
    r = h + Re
    g0 = Earth.g0 / params['scale'].accel

    f = np.zeros((5, 5))
    f[0, 1] = v * np.cos(gm) * Re / r ** 2
    f[0, 2] = -2 * mu / r ** 3 * np.sin(gm) - q * beta / (z * Bc) * (1 + sig ** 2 / 2)
    f[0, 3] = (v / r ** 2 - 2 * mu / r ** 3 / v) * np.cos(gm) + sig / v * LD * q * beta / (z * Bc)
    f[2, 0] = -np.sin(gm)
    f[2, 1] = - np.cos(gm) * Re / r
    f[2, 2] = 2 * q / (z * v * Bc) * (1 + sig ** 2 / 2)
    f[2, 3] = -(1 / r + mu / r ** 2 / v ** 2) * np.cos(gm) + ita * np.sin(delta) / (z * v ** 2) * tw * g0 + sig / (z * v ** 2) * LD * q / Bc
    f[3, 0] = -v * np.cos(gm)
    f[3, 1] = v * np.sin(gm) * Re / r
    f[3, 2] = mu / r ** 2 * np.cos(gm)
    f[3, 3] = (v / r - mu / (r ** 2 * v)) * np.sin(gm)
    f[4, 2] = ita * np.cos(delta) / z ** 2 * tw * g0 - q / (z ** 2 * Bc) * (1 + sig ** 2 / 2)
    f[4, 3] = ita * np.sin(delta) / (z ** 2 * v) * tw * g0 + sig / (z ** 2 * v) * LD * q / Bc
    return f


def StateDynamics(x, u, params):
    h, d, v, gm, z = x
    ita, delta, sig = u
    Bc = params['Bc']  # 弹道系数
    LD = params['LD']  # 最大升阻比
    tw = params['tw']  # 初始推重比
    Isp = params['Isp'] / params['scale'].time  # 比冲

    Re = Earth.radius / params['scale'].length  # 地球半径
    mu = Earth.mu / (params['scale'].length ** 3 / params['scale'].time ** 2)  # 地球引力常数
    beta = 1 / (Earth.H0 / params['scale'].length)  # 参考高度的倒数
    q = 1 / 2 * Earth.rho0 * np.exp(-h * beta) * v ** 2 * params['scale'].length  # 动压
    r = h + Re
    g0 = Earth.g0 / params['scale'].accel

    f = np.zeros((5,))
    f[0] = v * np.sin(gm)
    f[1] = v * np.cos(gm) * Re / r
    f[2] = -mu / r ** 2 * np.sin(gm) - q / (z * Bc) + ita * np.cos(delta) / z * tw * g0 - sig ** 2 / 2 * q / (z * Bc)
    f[3] = (v / r - mu / (r ** 2 * v)) * np.cos(gm) + ita * np.sin(delta) / z * tw * g0 + LD * sig / v * q / (z * Bc)
    f[4] = -ita * tw / Isp
    return f


def OptimalU(x, costate, params):
    h, d, v, gm, z = x
    ch, cd, cv, cgm, cz = costate
    LD = params['LD']  # 最大升阻比
    sigMax = params['sigMax']
    Isp = params['Isp'] / params['scale'].time  # 比冲
    g0 = Earth.g0 / params['scale'].accel

    cos_delta = -cv / np.sqrt(cv ** 2 + cgm ** 2 / v ** 2)
    sin_delta = -cgm / v / np.sqrt(cv ** 2 + cgm ** 2 / v ** 2)
    if sin_delta > 0:
        delta = np.arccos(cos_delta)
    else:
        delta = -np.arccos(cos_delta)

    s_throttle = cv * cos_delta + cgm / v * sin_delta - cz * z / (g0 * Isp)
    if s_throttle > 0:
        throttle = 0
    elif s_throttle < 0:
        throttle = 1
    else:
        throttle = 1 / 2  # TODO

    if cv == 0:
        sig = 0
    elif cv < 0:
        if cgm > 0:
            sig = 0
        elif sigMax / LD * cv * v <= cgm <= 0:
            sig = cgm * LD / (cv * v)
        else:
            sig = sigMax
    else:
        if cgm < sigMax * v / (2 * LD) * cv:
            sig = sigMax
        else:
            sig = 0
    return np.array([throttle, delta, sig])


def PSOFunc(dvs, **params):
    popsize = dvs.shape[0]
    res = np.zeros((popsize,))

    dt = params['scale'].sigmaScale
    for pp in range(popsize):
        state = dvs[pp, :5]
        costate = dvs[pp, 5:]
        stateHistory = [state]
        costateHistory = [costate]
        controlHistory = []
        Hval = []
        for step in range(10000):
            control = OptimalU(state, costate, params)
            dff = StateDynamics(state, control, params)
            dfc = CostateDynamics(state, control, params)
            state = state + dff * 1 / dt
            costate = costate + np.matmul(dfc, costate) * 1 / dt
            if np.any(state > params['xub']) or np.any(state < params['xlb']):
                break
            stateHistory.append(state)
            costateHistory.append(costate)
            controlHistory.append(control)
            Hval.append(np.dot(costate, dff))
        stateHistory = np.array(stateHistory)
        costateHistory = np.array(costateHistory)
        controlHistory = np.array(controlHistory)
        Hval = np.array(Hval)
        plotHistory(stateHistory, costateHistory, controlHistory, Hval, params)
    return res


def plotHistory(state, costate, control, Hval, params):
    plt.switch_backend('QT5Agg')
    state = state * params['scale'].stateScale.transpose()
    plt.figure(1)
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.plot(state[:, i])

    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(100, 100, 1200, 800)

    plt.figure(2)
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.plot(costate[:, i])
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(1280, 100, 1200, 800)

    plt.figure(3)
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(control[:, i])
    plt.subplot(2, 2, 4)
    plt.plot(Hval)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(1280, 100, 1200, 800)
    plt.show()


if __name__ == '__main__':
    planet = Earth()
    scale = Scale(length=planet.radius, velocity=planet.v0, mass=1,
                  state_name=['length', 'length', 'velocity', 'angle', 'non'],
                  control_name=['non', 'angle', 'non'])

    lb = np.array([1e3, 0, 100, -np.pi / 2, 0.1]) / scale.stateScale.squeeze()
    lb = np.concatenate([lb, -1000 * np.ones((5,))])
    ub = np.array([200e3, 1000e3, 5000, np.pi / 2, 1]) / scale.stateScale.squeeze()
    ub = np.concatenate([ub, 1000 * np.ones((5,))])

    bounds = (lb, ub)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # hyper-parameters
    kwargs = {'Bc': 1e3,
              'LD': 3,
              'sigMax': 3,
              'tw': 1.5,
              'Isp': 350,
              'scale': scale,
              'xlb': lb[:5],
              'xub': ub[:5]}
    # optimizer = GlobalBestPSO(n_particles=1000, dimensions=10, options=options, bounds=bounds)
    # cost, pos = optimizer.optimize(PSOFunc, iters=100, **kwargs)
    state0 = np.array([100e3, 0, 3000, np.deg2rad(10), 1]) / scale.stateScale.squeeze()
    costate0 = np.ones((5,)) * 0.1
    PSOFunc(np.concatenate([state0, costate0]).reshape((1, -1)), **kwargs)
