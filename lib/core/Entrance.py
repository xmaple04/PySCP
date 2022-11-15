"""
author: xmaple
date: 2021-11-06
entrance of PySCP algorithm
"""
import matplotlib.pyplot as plt
import numpy as np
from lib.utils.plotlib import *
from lib.core.MultiphaseSOCP import MultiPhaseSOCP
from lib.core.IterativeMethod import TrustRegion, TrustRegionLineSearch
from lib.utils.Structs import Result
from lib.utils.Structs import PhaseData
import warnings

warnings.filterwarnings('ignore')


class PySCP:
    def __init__(self, setup):
        """
        :param setup:
        setup['model']: model object.
        setup['scale']: auto scale or manual scale. if auto scale, the solver will automatically calculate appropriate scale for state, control and
            time; otherwise, the user must provide scale['state'], scale['control'] and scale['sigma'] to manually scale the quantities.
        """
        self.setup = self.preprocess(setup)
        self.model = self.setup['model']
        self.algorithm = self.setup['algorithm']  # 初始化算法参数
        self.socp = MultiPhaseSOCP(setup)  # 构造多段最优控制问题

        # 初始化结果变量
        self.result = Result()  # record results
        self.RecordResult()

    def solve(self, **kwargs):
        """ main function to solve the multi-phase ocp """
        maxiteration = self.setup['maxIteration']
        for step in range(maxiteration):
            if self.setup['verbose'] == 1:
                print('{}th in {} iteration'.format(step + 1, maxiteration))
            if self.setup['verbose'] == 2:
                print('-' * 100)
                print('{}th in {} iteration'.format(step + 1, maxiteration).center(100))
                print('-' * 100)

            # solve
            self.socp.solve(solver='ECOS',  verbose=False)
            # record
            self.RecordResult()
            self.result.addcvxtime(self.socp.cvxProb._solver_stats.solve_time)
            self.result.meshHistory.append(self.socp.gatherMeshes())
            self.result.stepNum += 1

            if not self.socp.isSolved():
                raise BaseException('There is no solution!')
            if self.socp.converged:
                break
            self.socp.update()

            # plot solving process
            if (step + 1) % self.setup['plot_interval'] == 0:
                self.plotXU(traj=self.result.solutionDimension, color='k', linewidth=0.6, **kwargs)

        # finalize
        self.postprocess()

    def getObjective(self):
        traj = self.result.solution
        obj = self.socp.getObjective(traj)
        return obj

    def plotPeroformanceHistory(self):
        print('Nonlinear:\t', self.algorithm.realJTrace)
        print('Linear:\t', self.algorithm.fakeJTrace)
        print('Ratio History:\t', self.algorithm.ratioHistory)

        plt.figure(figsize=(8, 6))
        ratios = np.zeros_like(self.algorithm.ratioHistory)
        ratios[0] = self.algorithm.ratioHistory[0]
        for i in range(1, len(self.algorithm.ratioHistory)):
            ratios[i] = ratios[i - 1] * self.algorithm.ratioHistory[i]
        plt.plot(np.arange(1, len(self.algorithm.ratioHistory) + 1), ratios)
        plt.xlabel('iteration')
        plt.ylabel('trust region radius')
        plt.savefig('trust region radius history.jpg', dpi=900, bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.plot(self.algorithm.realJTrace, marker='o')
        plt.plot(self.algorithm.fakeJTrace, marker='*')
        plt.plot(self.algorithm.allJTrace, marker='o')
        plt.plot(self.algorithm.fakeAllJTrace, marker='*')
        plt.show()

    def RecordResult(self, finish=False):
        self.result.solution = self.socp.gatherRefTrajectory()
        self.result.solutionDimension = self.socp.dimTrajectory(self.result.solution)
        self.result.errorHistory.append(self.socp.gatherDistDynamicsCost(self.result.solution))
        if finish:
            solutionIntegrated = self.socp.gatherIntegratedTrajectory()
            self.result.solutionIntegrated = self.socp.dimTrajectory(solutionIntegrated)

    def plotXU(self, traj, **kwargs):
        figsize = (10, 6)
        # pre-process
        if kwargs.get('matlab_path') is not None:
            matTraj = self.readMatlabData(kwargs.get('matlab_path'))

        # plot states in one figure, and controls in another
        tf = np.zeros((self.setup['phaseNum'],))
        for iPhase in range(self.setup['phaseNum']):
            if isinstance(traj[iPhase], PhaseData):
                xt = traj[iPhase].xtao
                state = traj[iPhase].state
                ut = traj[iPhase].utao
                control = traj[iPhase].control
            else:
                xt, state, ut, control = traj[iPhase]

            xdim = state.shape[1]
            udim = control.shape[1]
            xt = xt + tf[0]
            ut = ut + tf[0]
            tf[iPhase] = xt[-1]  # TODO

            stateyscale = np.ones((xdim,))
            controlyscale = np.ones((udim,))
            if kwargs.get('stateyscale') is not None:
                stateyscale = kwargs.get('stateyscale')
            if kwargs.get('controlyscale') is not None:
                controlyscale = kwargs.get('controlyscale')

            if kwargs.get('matlab_path'):
                mattime = matTraj[0]
                matState = matTraj[1]
                matControl = matTraj[2]

            if self.setup['plotStyle'] == 'grid':
                plt.figure(1, figsize=figsize)

            xrow = int(np.sqrt(xdim))
            xcol = np.ceil(xdim / xrow)
            urow = int(np.sqrt(udim))
            ucol = np.ceil(udim / urow)
            for ix in range(xdim):
                if self.setup['plotStyle'] == 'grid':
                    plt.subplot(xrow, xcol, ix + 1)
                else:
                    plt.figure(ix, figsize=figsize)
                plt.plot(xt, state[:, ix] * stateyscale[ix], color=kwargs.get('color'), marker=kwargs.get('marker'), linewidth=kwargs.get('linewidth'))

                if kwargs.get('matlab_path'):
                    plt.plot(mattime, matState[:, ix] * stateyscale[ix], color='r', linestyle='--', linewidth=kwargs.get('linewidth'))

                plt.xlabel('time/s')
                if kwargs.get('state_name'):
                    plt.ylabel(kwargs.get('state_name')[ix])
                else:
                    plt.ylabel('state {:d}'.format(ix))
                if kwargs.get('legend'):
                    plt.legend(kwargs.get('legend'))
            if kwargs.get('save'):
                plt.tight_layout()
                plt.savefig('state.eps', dpi=600)

            if self.setup['plotStyle'] == 'grid':
                plt.figure(2, figsize=figsize)
            for iu in range(udim):
                if self.setup['plotStyle'] == 'grid':
                    plt.subplot(urow, ucol, iu + 1)
                else:
                    plt.figure(xdim + iu, figsize=figsize)
                plt.plot(ut, control[:, iu] * controlyscale[iu], color=kwargs.get('color'), marker=kwargs.get('marker'))
                if kwargs.get('matlab_path'):
                    plt.plot(mattime, matControl[:, iu] * controlyscale[iu], color='r', linestyle='--')

                plt.xlabel('time/s')
                if kwargs.get('state_name'):
                    plt.ylabel(kwargs.get('control_name')[iu])
                else:
                    plt.ylabel('control {:d}'.format(iu))

                if kwargs.get('legend'):
                    plt.legend(kwargs.get('legend'))
            if kwargs.get('save'):
                plt.tight_layout()
                plt.savefig('control.eps', dpi=600)

        if kwargs.get('show'):
            plt.show()

    def readMatlabData(self, path):
        from scipy.io import loadmat

        data = loadmat(path)
        data = data['traj'][0][0]
        t = data['time']
        state = data['state']
        control = data['control']

        return t, state, control

    def print(self):
        from texttable import Texttable
        """ print solve information """
        tb = Texttable()
        tb.add_row(['Objective', self.result.objective])
        tb.add_row(['Maximum Error', self.result.maxError])
        tb.add_row(['Total Solving Time (s)', self.result.cvxTime])
        tb.add_row(['Solving Time Per Step (ms)', 1000 * (self.result.cvxTime / self.result.stepNum)])

        print('')
        print('Solution Information')
        print(tb.draw())

    def preprocess(self, setup):
        """
        预处理：无量纲化→
        preprocess before any operation starts. The workflow is nondimensionlize→
        :param setup:
        :return: setup
        """
        setup['phaseNum'] = len(setup['model'].phases)
        # 无量纲化处理, non-dimensionlize
        for phase in setup['model'].phases:
            phase.nonDimensionlize()

        # 显示级别，display level
        if 'verbose' in setup:
            assert setup['verbose'] in (0, 1, 2)
        else:
            setup['verbose'] = 2
        plot_interval = setup.get('plot_interval')
        if plot_interval is None:
            setup['plot_interval'] = 1000

        if 'adaptive' not in setup:
            setup['adaptive'] = False
        if 'tolerance' not in setup:
            setup['tolerance'] = 1e-4

        if isinstance(setup['meshConfig'], dict):
            setup['meshConfig'] = [setup['meshConfig']] * setup['phaseNum']

        # initialization method.
        # 初始化目前有两种方法：
        #   线性初始化：初始与终端时刻线性插值
        #   积分初始化：假设控制变量，对状态变量进行积分
        if 'initialization' not in setup:
            setup['initialization'] = 'linear'

        algorithm_flag = setup.get('algorithm')  # iterative algorithm: Trust region method or line search.
        if algorithm_flag is None or algorithm_flag == 'TrustRegion':
            algorithm = TrustRegion(setup)
            algorithm.setupTrustRegion(setup['model'].phases)
        else:
            algorithm = TrustRegionLineSearch(setup)  # TODO
        setup['algorithm'] = algorithm

        if not 'plotStyle' in setup:
            setup['plotStyle'] = 'grid'

        return setup

    def postprocess(self):
        self.RecordResult(finish=True)
        # self.result.maxError = self.getMaxRelativeError()
        self.result.maxRungeError = 0
        self.result.objective = self.getObjective()
