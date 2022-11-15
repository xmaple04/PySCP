"""
author: xmaple
date: 2021-11-02
给离散方法套个壳，以统一调用接口
"""
from lib.Mesh.MeshSS import ZOH, FOH, RungeKutta, SingleStep
from lib.Mesh.MeshHP import hpLG, hpLGR, hpfLGR, hpPseudoSpectral

method_dict = {'ZOH': ZOH,
               'FOH': FOH,
               'RK': RungeKutta,
               'LG': hpLG,
               'LGR': hpLGR,
               'fLGR': hpfLGR,
               'hpLG': hpLG,
               'hpLGR': hpLGR,
               'hpfLGR': hpfLGR}


class MeshShell:
    def __init__(self, setup):
        self.discrete = setup['discrete']
        method = method_dict[self.discrete]
        if issubclass(method, hpPseudoSpectral):
            """ pseudospectral methods """
            if isinstance(setup['degree'], list):
                degree = setup['degree']
            else:
                degree = [setup['degree']]
            self.mesh = method(degree=degree, seg_fractions=setup.get('seg_fractions'), xdim=setup['xdim'], udim=setup['udim'],
                               dynamics=setup['dynamics'])
        else:
            """ single step methods """
            self.mesh = method(num=setup.get('num'), mps=setup.get('mps'), xdim=setup['xdim'], udim=setup['udim'], dynamics=setup['dynamics'])

        # collection
        self.mps = self.mesh.mps  # mesh points

        # state variable and control variable
        self.xdim = setup['xdim']  # state variable number
        self.udim = setup['udim']  # control variable number
        self.nx = self.mesh.nx  # discrete state variable
        self.nu = self.mesh.nu  # discrete control variable

        # collocation for dynamic equation and path constraints
        self.cpNum = self.mesh.numCps
        self.cpState = self.mesh.cpState
        self.cpControl = self.mesh.cpControl

        # integral
        self.weights4State = self.mesh.weights4State  # integral weights for state
        self.weights4Control = self.mesh.weights4Control  # integral weights for control

        if self.isPseudoSpectral():
            self.PDM = self.mesh.PDM
            self.PIM = self.mesh.PIM
            self.PIMX0Index = self.mesh.PIMX0Index
            self.PIMXfIndex = self.mesh.PIMXfIndex

    def isPseudoSpectral(self):
        if issubclass(method_dict[self.discrete], hpPseudoSpectral):
            return True
        else:
            return False

    def isZOH(self):
        if self.discrete == 'ZOH':
            return True
        else:
            return False

    def refreshMatrices(self, xk, uk, sigmak, auxdata):
        """ refresh approximation matrices: A_tilde, B_tilde, B1_tilde, F_tilde, R_tilde """
        self.mesh.refreshMatrices(xk, uk, sigmak, auxdata)

    def getMatrices(self):
        """ get approximation matrices: A_tilde, B_tilde, B2_tilde, F_tilde, R_tilde """
        return self.mesh.A_tilde, self.mesh.B_tilde, self.mesh.B2_tilde, self.mesh.f_tilde, self.mesh.R_tilde

    def getNuDynamics(self, scheme, X, U, sigma):
        if scheme == 'differential':  # dynamics violation in pseudospectral differential form
            nu = self.mesh.DifferentialFormError(X, U, sigma)
        elif scheme == 'integral':  # dynamics violation in pseudospectral integral form
            nu = self.mesh.IntegralFormError(X, U, sigma)
        else:  # dynamics violation in single step form
            nu = self.mesh.getSingleStepError(X, U, sigma)
            Exception('scheme error')
        return nu

    def getPIMindices(self):
        """
         积分形式: x[PIMedIndex] = x0[PIMIndex] + I*fx， PIMIndex和PIMedIndex分别是积分起始序号和被积分位置序号
         In the integral form: x[PIMedIndex] = x0[PIMIndex] + I*fx, PIMIndex and PIMedIndex are the indices of integration starting states and the
         indices of the integrated states.
        """
        return self.mesh.PIMX0Index, self.mesh.PIMXfIndex
