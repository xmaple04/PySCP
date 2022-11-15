"""
author: xmaple
date: 2021-11-06
"""
import matplotlib.pyplot as plt
import numpy as np
from lib.core.Entrance import PySCP
from lib.utils.PlanetData import Earth, Moon
from lib.utils.Scale import Scale
from lib.utils.Structs import PhaseInfo, Linkage


class moonLander:
    def __init__(self):
        # scale
        scale = Scale(state_name=['non', 'non'], control_name=['non'])
        phase = PhaseInfo()
        phase.init_state_bound = np.array([[10, -2], [10, -2]])
        phase.state_bound = np.array([[0, -10], [20, 10]])
        phase.final_state_bound = np.array([[0, 0], [0, 0]])
        phase.control_bound = np.array([[0], [3]])
        phase.t0_bound = np.array([0, 0])
        phase.tf_bound = np.array([4, 5])
        phase.dynamicsFunc = self.Dynamics
        phase.pathFunc = self.Path
        phase.scale = scale
        phase.trState = np.ones((2,)) * 1
        phase.trControl = np.ones((1,)) * 1
        phase.trSigma = 1
        self.phases = [phase]
        self.linkages = []

    def Dynamics(self, x, u, t, auxdata):
        h, v = x
        thrust_acceleration = u[0]
        f = np.zeros_like(x)
        f[0] = v
        f[1] = -1.6 + thrust_acceleration

        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])
        return f, A, B

    def Path(self, phases):
        path = None
        return [path]

    def objective(self, Vars, refTraj, weights4State, weights4Control):
        obj = weights4Control[0] @ ((refTraj[0].sigma * Vars[0].control[:, 0]) + \
                                    (Vars[0].sigma * refTraj[0].control[:, 0]) - \
                                    (refTraj[0].sigma * refTraj[0].control[:, 0]))
        return obj


def paper():
    vehicle = moonLander()

    PSConfig = {'meshName': 'LGR',
                'scheme': 'integral',  # difference or integration
                'degree': [6] * 20}
    setup = {'model': vehicle,
             'meshConfig': PSConfig,
             'initialization': 'linear',
             'maxIteration': 10,
             'verbose': 2,
             'weightNu': [1e2],
             'weightTrustRegion': [5e-4]}

    prob = PySCP(setup)
    prob.solve()
    prob.plotXU(traj=prob.result.solutionIntegrated, show=False)
    prob.plotXU(traj=prob.result.solutionDimension,
                show=True,
                save=False,
                matlab_path='moonLanding.mat',
                state_name=[r'$h$', r'$v$'],
                control_name=['control'],
                legend=['Initial Guess', 'PySCP', 'GPOPS II'])
    prob.print()


def singleCase(setup):
    prob = PySCP(setup)
    prob.solve()
    error = prob.result.maxError
    cputime = prob.result.cvxTime * 1000
    obj = prob.result.objective
    return obj, cputime, error


def multipleCases():
    ssList = np.linspace(10, 100, 10).astype(np.int)
    ppList = np.linspace(10, 50, 10).astype(np.int)
    num = len(ssList)

    cpuTimes = np.zeros((num,))
    objs = np.zeros((num,))
    errors = np.zeros((num, 2))

    vehicle = moonLander()
    for i in range(num):
        PSConfig = {'meshName': 'LGR',
                    'scheme': 'differential',  # difference or integration
                    'degree': [ppList[i]] * 1,
                    'segFractions': [1. / 1] * 1}
        SSConfig = {'meshName': 'RK',
                    'ncp': ssList[i]}
        setup = {'model': vehicle,
                 'meshConfig': PSConfig,
                 'maxIteration': 15,
                 'verbose': 0}
        obj, cputime, error = singleCase(setup)
        cpuTimes[i] = cputime
        objs[i] = obj
        errors[i] = error[0]

    print('目标：', objs)
    bestObj = optimalValue()
    objError = (np.abs(np.array(objs) - bestObj) / bestObj)
    print('目标相对误差', objError)
    print('用时：', cpuTimes)
    print('误差：', errors)

    plt.figure()
    plt.plot(ssList, errors[:, 0], marker='o')
    plt.plot(ssList, errors[:, 1], marker='x')
    plt.yscale('log')
    plt.show()


def compare():
    ssList = 10 * np.arange(1, 11)
    ppList = 10 * np.arange(1, 11)
    hhList = np.arange(1, 11)
    num = len(ssList)

    cpuTimes = np.zeros((num,))
    objs = np.zeros((num,))
    errors = np.zeros((num, 2))
    markers = 'xos^*<^*<'

    plt.figure()
    vehicle = moonLander()
    labels = ['ZOH', 'FOH', 'RK', 'p-LG', 'p-LGR', 'p-LGR', 'h-LG', 'h-LGR', 'h-fLGR']

    for imethod in range(9):
        for i in range(num):
            method1 = {'meshName': 'ZOH', 'ncp': ssList[i]}
            method2 = {'meshName': 'FOH', 'ncp': ssList[i]}
            method3 = {'meshName': 'RK', 'ncp': ssList[i]}
            method4 = {'meshName': 'LG', 'scheme': 'integral', 'degree': [ppList[i]] * 1, 'segFractions': [1. / 1] * 1}
            method5 = {'meshName': 'LGR', 'scheme': 'integral', 'degree': [ppList[i]] * 1, 'segFractions': [1. / 1] * 1}
            method6 = {'meshName': 'fLGR', 'scheme': 'integral', 'degree': [ppList[i]] * 1, 'segFractions': [1. / 1] * 1}
            method7 = {'meshName': 'LG', 'scheme': 'integral', 'degree': [10] * hhList[i], 'segFractions': [1. / hhList[i]] * hhList[i]}
            method8 = {'meshName': 'LGR', 'scheme': 'integral', 'degree': [10] * hhList[i], 'segFractions': [1. / hhList[i]] * hhList[i]}
            method9 = {'meshName': 'fLGR', 'scheme': 'integral', 'degree': [10] * hhList[i], 'segFractions': [1. / hhList[i]] * hhList[i]}
            methods = [method1, method2, method3, method4, method5, method6, method7, method8, method9]
            setup = {'model': vehicle,
                     'meshConfig': methods[imethod],
                     'maxIteration': 15,
                     'verbose': 0}
            obj, cputime, error = singleCase(setup)
            cpuTimes[i] = cputime
            objs[i] = obj
            errors[i] = error[0]

        print('目标：', objs)
        bestObj = optimalValue()
        objError = (np.abs(np.array(objs) - bestObj) / bestObj)
        print('目标相对误差', objError)
        print('用时：', cpuTimes)
        print('误差：', errors)

        plt.figure(1)
        plt.plot(ssList, np.max(errors, axis=1), marker=markers[imethod], label=labels[imethod])

        plt.figure(2)
        plt.plot(ssList, objError, marker=markers[imethod], label=labels[imethod])

    plt.figure(1, figsize=(8, 6))
    plt.xticks(ssList)
    plt.yscale('log')
    plt.legend(loc="upper right")
    plt.xlabel('Node Number')
    plt.ylabel('State Error')
    plt.tight_layout()
    plt.savefig('Integration Error.eps', dpi=600)

    plt.figure(2, figsize=(8, 6))
    plt.yscale('log')
    plt.xticks(ssList)
    plt.legend()
    plt.xlabel('Node Number')
    plt.ylabel('Objective Error')
    plt.tight_layout()
    plt.savefig('Objective Error.eps', dpi=600)
    plt.show()


def optimalValue():
    g = 1.6
    a = 3 * g
    b = 12
    c = 20 * g - 56

    t1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    t2 = (2 + g * t1) / (3 - g)
    v1 = 2 + g * t1
    s = ((2 + v1) * t1 / 2) + v1 * t2 / 2
    # print('t1:{}'.format(t1))
    # print('t2:{}'.format(t2))
    # print(2 + g * t1 - (3 - g) * t2)
    # print(s)
    # print('optimal objective:{}'.format(3 * t2))
    return 3 * t2


if __name__ == '__main__':
    # optimalValue()
    paper()
    # compare()
