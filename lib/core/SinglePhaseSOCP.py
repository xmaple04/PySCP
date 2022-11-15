"""
author: xmaple
date: 2021-11-10
以setup开头的函数，都是为了建立cvxpy凸问题
"""
import cvxpy as cvx
import numpy as np
from lib.core.SinglePhaseManager import PhaseManager
from lib.utils.Structs import PhaseData, ParamStruct


class SingPhaseSOCP:
    def __init__(self, iPhase, setup):
        self.iPhase = iPhase  # phase number
        self.phaseInfo = setup['model'].phases[iPhase]  # phase information
        self.freeTime = self.phaseInfo.isFreeTime()  # free final time or fixed final time
        self.initialization = setup['initialization']
        self.algorithm = setup['algorithm']  # this is a reference object from MultiPhaseSOCP

        # mesh initialization
        self.xdim, self.udim = self.phaseInfo.getDimension()
        self.manager = PhaseManager(setup['model'], self.iPhase, setup['meshConfig'][iPhase])
        self.adaptive = setup['adaptive']

        self.dv = self.updateVars()
        # parameters initialization
        self.params = ParamStruct()
        self.updateParams(refTraj=self.getInitTrajectory())

    def update(self):
        if self.adaptive:
            self.updateMesh()
            self.updateVars()
        self.updateParams(refTraj=self.getNewTrajectory())

    def updateParams(self, refTraj):
        """
        根据参考轨迹更新近似矩阵
        update the approximation matrices around the reference trajectory.
        :return: None
        """
        self.params.ncp = self.manager.ncp
        self.params.integralWeight4State = self.manager.mesh.weights4State
        self.params.integralWeight4Control = self.manager.mesh.weights4Control

        self.params.refTraj = refTraj
        self.manager.refreshMatrices(refTraj.state, refTraj.control, refTraj.sigma, self.phaseInfo.auxdata)
        self.params.A_tilde, self.params.B_tilde, self.params.B2_tilde, self.params.F_tilde, self.params.R_tilde = self.manager.getApproximateMatrices()
        self.params.tr_state = (self.phaseInfo.state_bound[1] - self.phaseInfo.state_bound[0]) * self.algorithm.trState[self.iPhase] / 2
        self.params.tr_control = (self.phaseInfo.control_bound[1] - self.phaseInfo.control_bound[0]) * self.algorithm.trControl[self.iPhase] / 2
        if self.freeTime:
            self.params.tr_sigma = (np.mean(self.phaseInfo.tf_bound) - np.mean(self.phaseInfo.t0_bound)) * self.algorithm.trSigma[self.iPhase] / 2

    def updateVars(self):
        """ if mesh is adaptive, the cvxpy variables should be updated at every iteration """
        dv = PhaseData()
        dv.setupDVs(self.manager, self.freeTime, sigma=self.phaseInfo.getAverageSigma(), nPath=self.phaseInfo.nPath, nBoundary=self.phaseInfo.nBoundary)  # TODO: 目前只能处理初始时间为0的情况
        return dv

    # ----------------------------------------- setup functions ----------------------------------------- #
    # ------------------------------------------- 构造SOCP问题 -------------------------------------------- #
    def setupConstraints(self):
        """ setup constraints: including state bound, control bound, sigma bound,
         dynamics constraints, trust region constraints, path constraints """
        cstrs = []
        cstrs += self.setupStateConstraints()  # 状态变量约束, state variables constraints
        cstrs += self.setupControlConstraints()  # 控制变量约束, control variables constraints
        cstrs += self.setupTimeConstraints()

        if self.phaseInfo.nPath != 0:
            cstrs += self.setuPathConstraints()
        if self.phaseInfo.nBoundary != 0:
            cstrs += self.setupBoundaryConstraints()
        if self.algorithm.isTrustRegion():
            cstrs += self.setupTrustRegionConstraints()  # 信赖域约束, trust region constraints

        if self.manager.isPseudospectral():
            if self.manager.scheme == 'differential':  # 伪谱微分型式, pseudo-spectral differential form
                cstrs += self.setupPSDifferentialDynamics()
            elif self.manager.scheme == 'integral':  # 伪谱积分型式, pseudo-spectral integral form
                cstrs += self.setupPSIntegralDynamics()
        elif self.manager.isSingleStep():
            # single step discretization only has integral form
            cstrs += self.setupSSDynamics()
        elif self.manager.isHybrid():
            cstrs += self.setupHybridDynamics()
        elif self.manager.isTrapezoidal():
            cstrs += self.setupTrapezoidalDynamics()
        return cstrs

    # 动力学约束
    def setupDynamicViolation(self):
        """
        动力学约束违反，返回量为cvx变量，被添加到增强目标函数中
        dynamics violation.
        :return: return cvxpy variable, which will be added to the augmented objective function
        """
        return cvx.norm(self.dv.nuDynamics, "inf")
        # return cvx.sum(cvx.norm(self.dv.nuDynamics, axis=1))

    def setupPSIntegralDynamics(self):
        """
        Integral form for pseudo-spectral method
        """
        fx = []
        if self.freeTime:
            for k in range(self.params.ncp):
                fx.append(self.params.A_tilde[k] @ self.dv.state[self.manager.mesh.cpState[k]]
                          + self.params.B_tilde[k] @ self.dv.control[k]
                          + self.params.F_tilde[k] * self.dv.sigma
                          + self.params.R_tilde[k])
        else:
            for k in range(self.params.ncp):
                fx.append(self.params.A_tilde[k] @ self.dv.state[self.manager.mesh.cpState[k]]
                          + self.params.B_tilde[k] @ self.dv.control[k]
                          + self.params.F_tilde[k] * self.params.refTraj.sigma
                          + self.params.R_tilde[k])
        PIMedX = self.dv.state[self.manager.mesh.PIMX0Index] + self.manager.mesh.PIM @ cvx.vstack(fx)
        cstrns = [cvx.abs(self.dv.state[self.manager.mesh.PIMXfIndex] - PIMedX) <= self.dv.nuDynamics]
        return cstrns

    def setupPSDifferentialDynamics(self):
        """
        Differential form for pseudo-spectral method
        """
        cstrns = []
        if self.freeTime:
            for k in range(self.params.ncp):
                cstrns += [cvx.abs(self.manager.mesh.PDM[k] @ self.dv.state -
                                   (self.params.A_tilde[k] @ self.dv.state[self.manager.mesh.cpState[k]]
                                    + self.params.B_tilde[k] @ self.dv.control[k]
                                    + self.params.F_tilde[k] * self.dv.sigma
                                    + self.params.R_tilde[k]))
                           <= self.dv.nuDynamics[k]]
        else:
            for k in range(self.params.ncp):
                cstrns += [cvx.abs(self.manager.mesh.PDM[k] @ self.dv.state
                                   - (self.params.A_tilde[k] @ self.dv.state[self.manager.mesh.cpState[k]]
                                      + self.params.B_tilde[k] @ self.dv.control[k]
                                      + self.params.F_tilde[k] * self.params.refTraj.sigma
                                      + self.params.R_tilde[k]))
                           <= self.dv.nuDynamics[k]]
        return cstrns

    def setupSSDynamics(self):
        cstrns = []
        if self.freeTime:
            if self.manager.meshName == 'ZOH':
                for k in range(self.params.ncp):
                    cstrns += [cvx.abs(self.dv.state[k + 1]
                                       - (self.params.A_tilde[k] @ self.dv.state[k]
                                          + self.params.B_tilde[k] @ self.dv.control[k]
                                          + self.params.F_tilde[k] * self.dv.sigma
                                          + self.params.R_tilde[k]))
                               <= self.dv.nuDynamics[k]]
            else:
                for k in range(self.params.ncp):
                    cstrns += [cvx.abs(self.dv.state[k + 1]
                                       - (self.params.A_tilde[k] @ self.dv.state[k]
                                          + self.params.B_tilde[k] @ self.dv.control[k]
                                          + self.params.B2_tilde[k] @ self.dv.control[k + 1]
                                          + self.params.F_tilde[k] * self.dv.sigma
                                          + self.params.R_tilde[k]))
                               <= self.dv.nuDynamics[k]]
        else:
            if self.manager.meshName == 'ZOH':
                for k in range(self.params.ncp):
                    cstrns += [cvx.abs(self.dv.state[k + 1]
                                       - (self.params.A_tilde[k] @ self.dv.state[k]
                                          + self.params.B_tilde[k] @ self.dv.control[k]
                                          + self.params.F_tilde[k] * self.params.refTraj.sigma
                                          + self.params.R_tilde[k]))
                               <= self.dv.nuDynamics[k]]
            else:
                for k in range(self.params.ncp):
                    cstrns += [cvx.abs(self.dv.state[k + 1]
                                       - (self.params.A_tilde[k] @ self.dv.state[k]
                                          + self.params.B_tilde[k] @ self.dv.control[k]
                                          + self.params.B2_tilde[k] @ self.dv.control[k + 1]
                                          + self.params.F_tilde[k] * self.params.refTraj.sigma
                                          + self.params.R_tilde[k]))
                               <= self.dv.nuDynamics[k]]
        return cstrns

    def setupHybridDynamics(self):
        """
        Integral form for pseudo-spectral method
        """
        cstrns = []
        if self.freeTime:
            imat = 0
            for iseg in range(self.manager.mesh.nseg):
                segXk = self.dv.state[self.manager.mesh.segStateIndex[iseg]]
                segUk = self.dv.control[self.manager.mesh.segControlIndex[iseg]]

                if self.manager.mesh.segNames[iseg] == 'RK':
                    for k in range(self.manager.mesh.segDegrees[iseg]):
                        cstrns += [cvx.abs(segXk[k + 1]
                                           - (self.params.A_tilde[imat] @ segXk[k]
                                              + self.params.B_tilde[imat] @ segUk[k]
                                              + self.params.B2_tilde[imat] @ segUk[k + 1]
                                              + self.params.F_tilde[imat] * self.dv.sigma
                                              + self.params.R_tilde[imat]))
                                   <= self.dv.nuDynamics[imat]]
                        imat += 1
                else:
                    sfx = []
                    startIndex = imat
                    for k in range(self.manager.mesh.segDegrees[iseg]):
                        sfx.append(self.params.A_tilde[imat] @ segXk[k + 1]
                                   + self.params.B_tilde[imat] @ segUk[k]
                                   + self.params.F_tilde[imat] * self.dv.sigma
                                   + self.params.R_tilde[imat])
                        imat += 1
                    endIndex = imat
                    PIMX0Index = self.manager.mesh.PIMX0Indices[iseg]
                    PIMXfIndex = self.manager.mesh.PIMXfIndices[iseg]
                    PIM = self.manager.mesh.PIMs[iseg]
                    PIMedX = segXk[PIMX0Index] + PIM @ cvx.vstack(sfx)
                    cstrns += [cvx.abs(segXk[PIMXfIndex] - PIMedX) <= self.dv.nuDynamics[startIndex:endIndex]]
        else:
            for k in range(self.params.ncp):
                cstrns.append(self.params.A_tilde[k] @ self.dv.state[self.manager.mesh.cpState[k]]
                              + self.params.B_tilde[k] @ self.dv.control[k]
                              + self.params.F_tilde[k] * self.params.refTraj.sigma
                              + self.params.R_tilde[k])
        return cstrns

    def setupTrapezoidalDynamics(self):
        cstrns = []
        if self.freeTime:
            for k in range(self.params.ncp):
                cstrns += [cvx.abs(self.dv.state[k + 1] - self.dv.state[k]
                                   - (self.params.F_tilde[k] * self.dv.sigma
                                      + self.params.refTraj.sigma * self.params.B_tilde[k] @ self.dv.control[k]
                                      + self.dv.sigma * self.params.B_tilde[k] @ self.params.refTraj.control[k]
                                      - self.params.refTraj.sigma * self.params.B_tilde[k] @ self.params.refTraj.control[k]) / 2
                                   - (self.params.F_tilde[k + 1] * self.dv.sigma
                                      + self.params.refTraj.sigma * self.params.B_tilde[k + 1] @ self.dv.control[k + 1]
                                      + self.dv.sigma * self.params.B_tilde[k + 1] @ self.params.refTraj.control[k + 1]
                                      - self.params.refTraj.sigma * self.params.B_tilde[k + 1] @ self.params.refTraj.control[k + 1]) / 2)
                           <= self.dv.nuDynamics[k]]
        else:
            for k in range(self.params.ncp):
                cstrns += [cvx.abs(self.dv.state[k + 1] - self.dv.state[k]
                                   - self.params.refTraj.sigma * (self.params.F_tilde[k] + self.params.B_tilde[k] @ self.dv.control[k]) / 2
                                   - self.params.refTraj.sigma * (self.params.F_tilde[k + 1] + self.params.B_tilde[k + 1] @ self.dv.control[k + 1]) / 2)
                           <= self.dv.nuDynamics[k]]

        return cstrns

    # 路径约束
    def setupPathViolation(self):
        """
        动力学约束违反，返回量为cvx变量，被添加到增强目标函数中
        dynamics violation.
        :return: return cvxpy variable, which will be added to the augmented objective function
        """
        return cvx.norm(self.dv.nuPath, "inf")
        # return cvx.sum(cvx.norm(self.dv.nuPath, axis=1))

    def setuPathConstraints(self):
        """ path Constraints """
        cstr = []
        path = self.phaseInfo.pathFunc(self.dv, self.getRefTrajectory(), self.manager.mesh.cpState, self.manager.mesh.cpControl, self.phaseInfo.auxdata)
        for ic in range(self.phaseInfo.nPath):
            cstr += [path[ic] <= self.dv.nuPath[:, ic]]
        return cstr

    # 边界约束
    def setupBoundaryViolation(self):
        """
        动力学约束违反，返回量为cvx变量，被添加到增强目标函数中
        dynamics violation.
        :return: return cvxpy variable, which will be added to the augmented objective function
        """
        return cvx.norm(self.dv.nuBoundary, "inf")
        # return cvx.sum(cvx.norm(self.dv.nuBoundary, axis=1))

    def setupBoundaryConstraints(self):
        """ path Constraints """
        cstr = []
        bound = self.phaseInfo.boundaryFunc(self.dv, self.getRefTrajectory(), self.phaseInfo.auxdata)
        for ic in range(self.phaseInfo.nBoundary):
            cstr += [bound[ic] <= self.dv.nuBoundary[ic]]
        return cstr

    # 固定约束
    def setupTrustRegionConstraints(self):
        """
        信赖域约束。TODO: 如果是仿射控制系统，是否不需要增加控制信赖域
        Trust region constraints.
        :return: constraints
        """
        cstrs = [
            cvx.abs(self.dv.state - self.params.refTraj.state) <= self.params.tr_state,
            cvx.abs(self.dv.control - self.params.refTraj.control) <= self.params.tr_control
        ]
        if self.freeTime:
            cstrs += [cvx.abs(self.dv.sigma - self.params.refTraj.sigma) <= self.params.tr_sigma]
        return cstrs

    def setupStateConstraints(self):
        cstrs = [self.dv.state[0] >= self.phaseInfo.init_state_bound[0],
                 self.dv.state[0] <= self.phaseInfo.init_state_bound[1],
                 self.dv.state[1:-1] >= self.phaseInfo.state_bound[0].reshape((1, -1)),
                 self.dv.state[1:-1] <= self.phaseInfo.state_bound[1].reshape((1, -1)),
                 self.dv.state[-1] >= self.phaseInfo.final_state_bound[0],
                 self.dv.state[-1] <= self.phaseInfo.final_state_bound[1]]
        return cstrs

    def setupControlConstraints(self):
        cstrs = [self.dv.control >= self.phaseInfo.control_bound[0].reshape((1, -1)),
                 self.dv.control <= self.phaseInfo.control_bound[1].reshape((1, -1))]
        return cstrs

    def setupTimeConstraints(self):
        if self.freeTime:
            sigma_min = (self.phaseInfo.tf_bound[0] - self.phaseInfo.t0_bound[1]) / 2  # TODO
            sigma_max = (self.phaseInfo.tf_bound[1] - self.phaseInfo.t0_bound[0]) / 2
            cstrs = [self.dv.sigma >= sigma_min,
                     self.dv.sigma <= sigma_max]
        else:
            cstrs = []
        return cstrs

    # ----------------------------------------- get trajectory ----------------------------------------- #
    # -------------------------------------------- 获取轨迹 --------------------------------------------- #
    def getInitTrajectory(self):
        traj = PhaseData()
        traj.xtao = self.manager.mesh.xtao
        traj.utao = self.manager.mesh.utao
        phaseInfo = self.phaseInfo

        if phaseInfo.guess_x0 is not None:
            x0 = phaseInfo.guess_x0
            xf = phaseInfo.guess_xf
            u0 = phaseInfo.guess_u0
            uf = phaseInfo.guess_uf
            t0 = phaseInfo.guess_t0
            tf = phaseInfo.guess_tf
        else:
            x0 = np.mean(phaseInfo.init_state_bound, axis=0)
            xf = np.mean(phaseInfo.final_state_bound, axis=0)
            u0 = phaseInfo.control_bound[0]
            uf = phaseInfo.control_bound[1]
            t0 = np.mean(phaseInfo.t0_bound)
            tf = np.mean(phaseInfo.tf_bound)

        if self.initialization == 'linear':
            traj.state = (1 - traj.xtao.reshape((-1, 1))) / 2 * x0.reshape((1, -1)) + (1 + traj.xtao.reshape((-1, 1))) / 2 * xf.reshape((1, -1))
            traj.control = (1 - traj.utao.reshape((-1, 1))) / 2 * u0.reshape((1, -1)) + (1 + traj.utao.reshape((-1, 1))) / 2 * uf.reshape((1, -1))
            traj.sigma = (tf - t0) / 2
        elif self.initialization == 'integration':  # 积分初始化
            traj.state = (1 - traj.xtao.reshape((-1, 1))) / 2 * x0.reshape((1, -1)) + (1 + traj.xtao.reshape((-1, 1))) / 2 * xf.reshape((1, -1))
            traj.control = (1 - traj.utao.reshape((-1, 1))) / 2 * u0.reshape((1, -1)) + (1 + traj.utao.reshape((-1, 1))) / 2 * uf.reshape((1, -1))
            traj.state[:, -1] = 0
            traj.control = traj.control * 0
            traj.sigma = (tf - t0) / 2
            traj = self.getPropagatedTrajectory(traj, continuous=True)
        elif self.initialization == 'specify':
            traj.state = traj.state * 0
            traj.control = traj.control * 0
            traj.sigma = (tf - t0) / 2

        return traj

    def getRefTrajectory(self):
        """ 当前步的参考轨迹， state, control, sigma """
        refTraj = self.params.refTraj
        return refTraj

    def getNewTrajectory(self):
        newTraj = self.dv.getPhaseValues()
        return newTraj

    def getDynamicsViolation(self, phaseTraj):
        """
        original function dynamics violation
        :param phaseTraj:
        :return:
        """
        pointcost = self.manager.getPointwiseDynamicsCost(phaseTraj.state, phaseTraj.control, phaseTraj.sigma, self.phaseInfo.auxdata)
        return pointcost

    def getPropagatedTrajectory(self, refTraj, continuous=False):
        from scipy.integrate import odeint
        from scipy.interpolate import interp1d

        def propagateFunc(x, tao, ufuncs, sigma):
            udim = len(ufuncs)
            control = np.ones((udim,))
            for iu in range(udim):
                control[iu] = ufuncs[iu](tao)
            dx, _, _ = self.phaseInfo.dynamicsFunc(x, control, 0, self.phaseInfo.auxdata)
            return sigma * dx

        """ 当前步的参考轨迹， state, control, sigma """
        trajInt = PhaseData()

        xtao = refTraj.xtao
        state = refTraj.state
        utao = refTraj.utao
        control = refTraj.control
        sigma = refTraj.sigma

        udim = control.shape[1]
        ufuncs = []
        for iu in range(udim):
            ufuncs.append(interp1d(utao, control[:, iu], kind='linear', fill_value='extrapolate'))

        stateInt = np.zeros_like(state)
        stateInt[0] = state[0]
        if continuous:
            for i in range(len(state) - 1):
                stateInt[i + 1] = odeint(propagateFunc, stateInt[i], [xtao[i], xtao[i + 1]], args=(ufuncs, sigma,))[1]
        else:
            for i in range(len(state) - 1):
                stateInt[i + 1] = odeint(propagateFunc, state[i], [xtao[i], xtao[i + 1]], args=(ufuncs, sigma,))[1]

        trajInt.xtao = xtao
        trajInt.utao = utao
        trajInt.state = stateInt
        trajInt.control = control
        trajInt.sigma = sigma
        return trajInt

    def getOpenLoopPropagatedTrajectory(self, refTraj):
        trajInt = self.getPropagatedTrajectory(refTraj, continuous=True)
        return trajInt

    # ----------------------------------------- violations ----------------------------------------- #
    # --------------------------------------- SOCP约束违反度 ----------------------------------------- #
    def getNonlinearDynamicsCost(self, phaseTraj):
        """
        original function dynamics violation
        :param phaseTraj:
        :return:
        """
        distCost = self.getDistNonlinearDynamicsCost(phaseTraj)
        cost = np.max(np.linalg.norm(distCost, ord=1, axis=1))
        # cost = np.sum(np.linalg.norm(distCost, axis=1))
        return cost

    def getDistNonlinearDynamicsCost(self, phaseTraj):
        propagetedTraj = self.getPropagatedTrajectory(phaseTraj, continuous=False)
        stateInt = propagetedTraj.state
        state = phaseTraj.state
        distCost = np.abs(stateInt - state)

        # distCost = self.getDynamicsViolation(phaseTraj)
        return distCost

    def getNonlinearPathCost(self, phaseTraj):
        """
        original function dynamics violation
        :param phaseTraj:
        :return:
        """
        if self.phaseInfo.nPath:
            nu = self.phaseInfo.pathFunc(phaseTraj, self.getRefTrajectory(), self.manager.mesh.cpState, self.manager.mesh.cpControl, self.phaseInfo.auxdata)
            nu = np.clip(nu, 0., None)
            cost = np.max(np.linalg.norm(nu, ord=1, axis=1))
            # cost = np.sum(np.linalg.norm(nu, axis=1))
            return cost
        else:
            return 0.

    def getNonlinearBoundaryCost(self, phaseTraj):
        if self.phaseInfo.nBoundary:
            nu = self.phaseInfo.boundaryFunc(phaseTraj, self.getRefTrajectory(), self.phaseInfo.auxdata)
            nu = np.clip(nu, 0., None)
            cost = np.max(np.linalg.norm(nu, ord=1, axis=1))
            # cost = np.sum(np.linalg.norm(nu, axis=1))
            return cost
        else:
            return 0.

    # ----------------------------------------- 其他 ----------------------------------------- #
    def getStepLength(self):
        # TODO: sigma = tf/2, t0 = 0
        # traj距离参考轨迹refTraj的距离，以变量取值范围为单位1
        refTraj = self.params.refTraj
        newTraj = self.getNewTrajectory()
        dx = (newTraj.state - refTraj.state) / (self.phaseInfo.state_bound[1] - self.phaseInfo.state_bound[0])
        dx = np.linalg.norm(dx, ord=np.inf, axis=0)
        du = (newTraj.control - refTraj.control) / (self.phaseInfo.control_bound[1] - self.phaseInfo.control_bound[0])
        du = np.linalg.norm(du, ord=np.inf, axis=0)
        ds = (newTraj.sigma - refTraj.sigma) / ((self.phaseInfo.tf_bound[1] - self.phaseInfo.tf_bound[0]) / 2)
        return dx, du, ds

    def updateMesh(self):
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

        def plotProcess():
            import matplotlib.pyplot as plt
            plt.figure()
            num = len(self.trial_error)
            ratios = []
            for i in range(num):
                ratio = (self.trial_error[i] / self.origin_error[i])
                ratios.append(ratio)
            ratios = np.array(ratios)
            width = 0.2
            for i in range(ratios.shape[1]):
                plt.bar(np.arange(num) + i * width, ratios[:, i], width / 2, color='salmon', label='类别A')
            plt.show()

        # if self.iPhase == 1:
        #     # --------------------------------- 误差 --------------------------------- #
        #     refTraj = self.getRefTrajectory()  # 当前轨迹
        #     cost = self.manager.getPointwiseDynamicsCost(refTraj.state, refTraj.control, refTraj.sigma, self.phaseInfo.auxdata)
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     plt.plot(self.manager.mesh.xtao[1:], np.log(np.linalg.norm(cost, axis=1)))
        #     plt.show()

        # --------------------------------- 网格优化 --------------------------------- #
        # if self.remesh_count != 0:  # 如果尚不满足更新网格条件，则后续语句不再执行
        #     self.remesh_count -= 1
        #     return
        # hp = self.manager.hp
        # intervals = self.manager.intervals
        # ratio = seg_error_trial / seg_error_origin
        #
        # new_hp = []
        # new_intervals = []
        # if self.remesh_flag != 0:
        #     self.remesh_flag -= 1
        # for seg in range(len(ratio)):
        #     if ratio[seg] >= 5:
        #         new_hp.append(hp[seg])
        #         new_hp.append(hp[seg])
        #         new_intervals.append(intervals[seg] / 2)
        #         new_intervals.append(intervals[seg] / 2)
        #     else:
        #         new_hp.append(hp[seg])
        #         new_intervals.append(intervals[seg])
        #
        # if self.remesh_flag:  # 更新网格
        #     self.meshConfig['segFractions'] = [1]  # TODO
        #     self.manager = PhaseManager(self.meshConfig)
