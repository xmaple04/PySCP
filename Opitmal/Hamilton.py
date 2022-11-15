"""
author: xmaple
date: 2022-01-10
"""
"""
author: xmaple
date: 2021-11-18
"""
import numpy as np
from lib.core.Entrance import PySCP
from lib.utils.Scale import Scale
from lib.utils.Structs import PhaseInfo
from lib.utils.PlanetData import Earth

planet = Earth()


class AtmosphereFlight:
    def __init__(self):
        self.Bc = 1e3
        self.LD = 3
        self.sigMax = 3
        self.tw = 1.5
        self.Isp = 350
        self.eps = 0.5

        vmin = 50
        # scale
        scale = Scale(length=planet.radius, velocity=planet.v0, mass=1,
                      state_name=['length', 'length', 'velocity', 'angle', 'non'],
                      control_name=['non', 'angle', 'non'])
        self.scale = scale

        csmin = -1
        csmax = 1
        phase = PhaseInfo()
        phase.init_state_bound = np.array([[80e3, 0, 2000, np.deg2rad(0), 1, csmin, csmin, csmin, csmin, csmin],
                                           [120e3, 500e3, 5000, np.deg2rad(40), 1, csmax, csmax, csmax, csmax, csmax]])
        phase.state_bound = np.array([[1e3, 0, vmin, np.deg2rad(-92), self.eps, csmin, csmin, csmin, csmin, csmin],
                                      [150e3, 500e3, 4000, np.deg2rad(40), 1, csmax, csmax, csmax, csmax, csmax]])
        phase.final_state_bound = np.array([[1e3, 0, 100, np.deg2rad(-92), self.eps, csmin, csmin, csmin, csmin, csmin],
                                            [5e3, 500e3, 200, np.deg2rad(-88), 1, csmax, csmax, csmax, csmax, csmax]])
        phase.control_bound = np.array([[0, -np.pi / 2, 0], [1, np.pi / 2, self.sigMax]])
        phase.t0_bound = np.array([0, 0])
        phase.tf_bound = np.array([100, 500])
        phase.dynamicsFunc = self.Dynamics
        phase.pathFunc = self.Path
        phase.nPath = 0
        phase.scale = scale
        phase.trState = np.ones((10,)) * 1
        phase.trControl = np.ones((3,)) * 1
        phase.trSigma = 1
        self.phases = [phase]
        self.linkages = []
        self.phaseNum = len(self.phases)

    def Dynamics(self, x, u, t, auxdata):
        state = x[:5]
        costate = x[5:]

        f_state = self.StateDynamics(state, u)
        f_costate = self.CostateDynamics(state, u)
        f = np.zeros((10, 10))

        # approximated dynamic equation
        A = np.zeros((10, 10))
        A[:5, :5] = f_state
        A[5:, 5:] = f_costate

        B = np.array((10, 3))

        return f, A, B

    def CostateDynamics(self, x, u):
        h, d, v, gm, z = x
        ita, delta, sig = u
        Bc = self.Bc  # 弹道系数
        LD = self.LD  # 最大升阻比
        tw = self.tw  # 初始推重比

        Re = Earth.radius / self.scale.length  # 地球半径
        mu = Earth.mu / (self.scale.length ** 3 / self.scale.time ** 2)  # 地球引力常数
        beta = 1 / (Earth.H0 / self.scale.length)  # 参考高度的倒数
        q = 1 / 2 * Earth.rho0 * np.exp(-h * beta) * v ** 2 * self.scale.length  # 动压
        r = h + Re
        g0 = Earth.g0 / self.scale.accel

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

    def StateDynamics(self, x, u):
        h, d, v, gm, z = x
        ita, delta, sig = u
        Bc = self.Bc  # 弹道系数
        LD = self.LD  # 最大升阻比
        tw = self.tw  # 初始推重比
        Isp = self.Isp / self.scale.time  # 比冲

        Re = Earth.radius / self.scale.length  # 地球半径
        mu = Earth.mu / (self.scale.length ** 3 / self.scale.time ** 2)  # 地球引力常数
        beta = 1 / (Earth.H0 / self.scale.length)  # 参考高度的倒数
        q = 1 / 2 * Earth.rho0 * np.exp(-h * beta) * v ** 2 * self.scale.length  # 动压
        r = h + Re
        g0 = Earth.g0 / self.scale.accel

        f = np.zeros((5,))
        f[0] = v * np.sin(gm)
        f[1] = v * np.cos(gm) * Re / r
        f[2] = -mu / r ** 2 * np.sin(gm) - q / (z * Bc) + ita * np.cos(delta) / z * tw * g0 - sig ** 2 / 2 * q / (z * Bc)
        f[3] = (v / r - mu / (r ** 2 * v)) * np.cos(gm) + ita * np.sin(delta) / z * tw * g0 + LD * sig / v * q / (z * Bc)
        f[4] = -ita * tw / Isp
        return f

    def OptimalU(self, x, costate):
        h, d, v, gm, z = x
        ch, cd, cv, cgm, cz = costate
        LD = self.LD  # 最大升阻比
        sigMax = self.sigMax
        Isp = self.Isp / self.scale.time  # 比冲
        g0 = Earth.g0 / self.scale.accel

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

    def Path(self, phases):
        path = None
        return [path]

    def objective(self, Vars, refTraj, weights4State, weights4Control):
        obj = 0
        return obj


if __name__ == '__main__':
    vehicle = AtmosphereFlight()

    PSConfig = {'meshName': 'fLGR',
                'scheme': 'differential',  # difference or integration
                'adaptive': False,
                'degree': [40] * 2,
                'segFractions': [0.5] * 2,
                'tolerance': 1e-6,
                'maxDegree': 50,
                'maxIteration': 5}
    SSConfig = {'meshName': 'FOH',
                'adaptive': False,
                'ncp': 40,
                'tolerance': 1e-6,
                'maxDegree': 50,
                'maxIteration': 5}

    setup = {'model': vehicle,
             'meshConfig': SSConfig,
             'algorithm': 'TrustRegion',
             'initialization': 'linear',
             'verbose': 2}

    prob = PySCP(setup)
    prob.drawInit(show=False)
    prob.solve()
    prob.plotXU(show=True,
                save=False,
                matlab_path='breakwell.mat',
                state_name=['h', 'v'],
                control_name=['control'],
                legend=['Initial Guess', 'PySCP', 'GPOPS II'])
    prob.print()
