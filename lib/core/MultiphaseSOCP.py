"""
author: xmaple
date: 2021-10-25
"""
import cvxpy as cvx
import numpy as np
from .SinglePhaseSOCP import SingPhaseSOCP


class MultiPhaseSOCP:
    def __init__(self, setup):
        self.model = setup['model']
        self.phaseNum = setup['phaseNum']
        self.verbose = setup['verbose']
        self.algorithm = setup['algorithm']
        self.tolerance = setup['tolerance']
        self.converged = False
        self.phases = [SingPhaseSOCP(iPhase, setup) for iPhase in range(self.phaseNum)]

        if self.model.linkages:
            self.linkageCVXvar = cvx.Variable(nonneg=True)

        self.cvxProb = self.setupProblem()

        traj = self.gatherRefTrajectory()
        J_hat = self.getAugmentedObjective(traj)
        if self.verbose:
            print('|\tJ={:.8g}'.format(J_hat))
        self.algorithm.initJTrace(J_hat)

    def setupProblem(self):
        cstrs = []
        dvs = []
        for iPhase in range(self.phaseNum):
            dvs.append(self.phases[iPhase].dv)

        refTraj = self.gatherRefTrajectory()
        weights4State, weights4Control = self.gatherIntegralWeights()
        obj = self.model.objective(dvs, refTraj, weights4State, weights4Control)

        for iPhase in range(self.phaseNum):
            cstrs += self.phases[iPhase].setupConstraints()
            obj += self.algorithm.weightDynamics[iPhase] * self.phases[iPhase].setupDynamicViolation()  # dynamics violation
            if self.phases[iPhase].phaseInfo.nPath:
                obj += self.algorithm.weightPath[iPhase] * self.phases[iPhase].setupPathViolation()  # path penalty
            if self.phases[iPhase].phaseInfo.nBoundary:
                obj += self.algorithm.weightBoundary[iPhase] * self.phases[iPhase].setupBoundaryViolation()  # boundary penalty

        if self.model.linkages:
            cstrs += self.setupLinkages()
            obj += self.algorithm.weightLinkage * self.linkageCVXvar

        cvxProb = cvx.Problem(cvx.Minimize(obj), cstrs)
        # check
        if not cvxProb.is_dcp():
            print('The problem is not convex')
        return cvxProb

    def setupLinkages(self):
        cstrs = []
        for linkage in self.model.linkages:
            for istate in linkage.index:
                diff = linkage.diff
                difference = self.phases[linkage.right].dv.state[0, istate] * self.phases[linkage.right].phaseInfo.scale.stateScale[istate] - \
                             self.phases[linkage.left].dv.state[-1, istate] * self.phases[linkage.left].phaseInfo.scale.stateScale[istate]
                cstrs += [difference >= diff[0] - self.linkageCVXvar,
                          difference <= diff[1] + self.linkageCVXvar]
        return cstrs

    def update(self):
        self.cvxProb = self.setupProblem()  # TODO：自适应改变网格时再调用
        traj = self.gatherNewTrajectory()
        J_hat = self.getAugmentedObjective(traj)
        # refresh = self.algorithm.TrustRegionAdaption(J_hat)
        # if self.verbose:
        #     print('|\tJ={:.8g}'.format(J_hat))
        # if refresh:
        for iPhase in range(self.phaseNum):
            self.phases[iPhase].update()

    def recordHistory(self):
        print('信赖域：')
        print(self.algorithm.trState, self.algorithm.trControl, self.algorithm.trSigma)
        for iPhase in range(self.phaseNum):
            dx, du, ds = self.phases[iPhase].getStepLength()
            print('步长:\t', iPhase, dx, du, ds)

    def solve(self, **kwargs):
        try:
            self.cvxProb.solve(**kwargs)
        except Exception as e:
            print(e)

    def isSolved(self):
        if 'optimal' in self.cvxProb.status:
            return True
        else:
            return False

    def getAugmentedObjective(self, traj):
        from texttable import Texttable
        costDynamics = []
        costPath = []
        costBoundary = []
        violationLinkage = 0.

        tb = Texttable()
        tb.set_max_width(400)
        tb.add_row(['phase', 'dynamic', 'path', 'boundary'])
        tb.set_cols_align(['l', 'r', 'r', 'r'])
        tb.set_cols_dtype(['i', 'e', 'e', 'e'])

        for iPhase in range(self.phaseNum):
            phaseTraj = traj[iPhase]

            costDynamics.append(self.phases[iPhase].getNonlinearDynamicsCost(phaseTraj))
            costPath.append(self.phases[iPhase].getNonlinearPathCost(phaseTraj))
            costBoundary.append(self.phases[iPhase].getNonlinearBoundaryCost(phaseTraj))

        if self.verbose == 2:
            for iPhase in range(self.phaseNum):
                tb.add_row([iPhase,
                            costDynamics[iPhase],
                            costPath[iPhase],
                            costBoundary[iPhase]])

        for linkage in self.model.linkages:
            diffThresh = linkage.diff
            leftScale = np.squeeze(self.phases[linkage.right].phaseInfo.scale.stateScale[linkage.index])
            rightScale = np.squeeze(self.phases[linkage.left].phaseInfo.scale.stateScale[linkage.index])
            diff = traj[linkage.right].state[0, linkage.index] * leftScale / rightScale - \
                   traj[linkage.left].state[-1, linkage.index]
            violationLinkage += np.sum(np.abs(diff[np.where(diff <= diffThresh[0])])) + \
                                np.sum(np.abs(diff[np.where(diff >= diffThresh[1])]))

            if self.verbose == 2:
                print('linkage violation:\t{:.8f}'.format(violationLinkage))

        if self.model.linkages:
            nuLinkage = self.linkageCVXvar.value

        if self.verbose == 2:
            tb.align = 'r'
            tb.float_format = '.8'
            print(tb.draw())

        J_hat = self.getObjective(traj) \
                + np.dot(self.algorithm.weightDynamics, costDynamics) \
                + np.dot(self.algorithm.weightPath, costPath) \
                + np.dot(self.algorithm.weightBoundary, costBoundary) \
                + self.algorithm.weightLinkage * violationLinkage

        if np.sum(costDynamics) < self.tolerance and \
                np.sum(costPath) < self.tolerance and \
                np.sum(costBoundary) < self.tolerance and \
                violationLinkage < self.tolerance:
            if self.algorithm.JTrace:
                if abs(self.algorithm.JTrace[-1] - J_hat) < 1e-4:
                    self.converged = True

        return J_hat

    def getObjective(self, refTraj):
        dvs = []
        for iPhase in range(self.phaseNum):
            dvs.append(self.phases[iPhase].dv)
        weights4State, weights4Control = self.gatherIntegralWeights()
        obj = self.model.objective(refTraj, refTraj, weights4State, weights4Control)
        return obj

    # ----------------------------------------- gather functions ----------------------------------------- #
    def gatherMeshes(self):
        """ collect mesh of all phases """
        meshes = []
        for iPhase in range(self.phaseNum):
            meshes.append(self.phases[iPhase].manager)
        return meshes

    def gatherTrustRegionBoundary(self, variables, trust_region):
        """
        trust region lower bound and upper bound for all phases
        """
        upper = []
        lower = []
        for iPhase in range(self.phaseNum):
            phase_upper = []
            phase_lower = []
            for i in range(3):  # TODO
                phase_upper.append(variables[iPhase][i] + trust_region[iPhase][i])
                phase_lower.append(variables[iPhase][i] - trust_region[iPhase][i])
            upper.append(phase_upper)
            lower.append(phase_lower)
        return upper, lower

    def gatherDynamicsViolationDistribution(self, variables):
        """ dynamics violation at each collocation point in all phases """
        nvs = []
        for iPhase in range(self.phaseNum):
            nv = self.phases[iPhase].getNonlinearDynamicsCost(variables[iPhase])
            nvs.append(nv)
        return nvs

    def gatherRefTrajectory(self):
        """
        record the iterative solution process
        """
        refTraj = [self.phases[iPhase].params.refTraj for iPhase in range(self.phaseNum)]
        return refTraj

    def gatherRefTrajectoryDimensioned(self):
        """ reference trajectory of all phases (dimensioned) """
        refTraj = self.gatherRefTrajectory()
        return refTraj

    def gatherNewTrajectory(self):
        """ reference trajectory of all phases """
        refTraj = [self.phases[iPhase].getNewTrajectory() for iPhase in range(self.phaseNum)]
        return refTraj

    def dimTrajectory(self, trajectory):
        """ reference trajectory of all phases (dimensioned) """
        refTraj = [trajectory[iPhase].getDimensionlizedData(self.phases[iPhase].phaseInfo) for iPhase in range(self.phaseNum)]
        return refTraj

    def gatherIntegratedTrajectory(self):
        integratedTraj = [self.phases[iPhase].getOpenLoopPropagatedTrajectory(self.phases[iPhase].getRefTrajectory()) for iPhase in range(self.phaseNum)]
        return integratedTraj

    def gatherDistDynamicsCost(self, trajectory):
        distCosts = [self.phases[iPhase].getDistNonlinearDynamicsCost(trajectory[iPhase]) for iPhase in range(self.phaseNum)]
        return distCosts

    def gatherIntegralWeights(self):
        weights4State = [self.phases[iPhase].params.integralWeight4State for iPhase in range(self.phaseNum)]
        weights4Control = [self.phases[iPhase].params.integralWeight4Control for iPhase in range(self.phaseNum)]
        return weights4State, weights4Control
