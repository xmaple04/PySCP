"""
结构体
"""
import cvxpy as cvx
import numpy as np


class PhaseInfo:
    def __init__(self):
        """ 确保数据的维度是2×xdim和2×udim的 """
        self.init_state_bound = None
        self.state_bound = None
        self.final_state_bound = None
        self.control_bound = None
        self.t0_bound = None
        self.tf_bound = None
        self.scale = None

        self.guess_x0 = None
        self.guess_xf = None
        self.guess_u0 = None
        self.guess_uf = None
        self.guess_t0 = None
        self.guess_tf = None
        self.guess_x = None
        self.guess_u = None
        self.guess_sigma = None

        self.dynamicsFunc = None  # Dynamics function
        self.secondOrder = None
        self.pathFunc = None  # path constraints
        self.boundaryFunc = None  # boundary constraints
        self.nPath = 0
        self.nBoundary = 0

        self.trState = 1.
        self.trControl = 1.
        self.trSigma = 1.

        self.auxdata = {}
        self.dimensionFlag = True

    def nonDimensionlize(self):
        assert np.all(self.init_state_bound[0] >= self.state_bound[0])
        assert np.all(self.init_state_bound[0] <= self.state_bound[1])
        assert np.all(self.final_state_bound[0] >= self.state_bound[0])
        assert np.all(self.final_state_bound[0] <= self.state_bound[1])
        if self.dimensionFlag:
            self.init_state_bound = self.init_state_bound / self.scale.stateScale.flatten()
            self.state_bound = self.state_bound / self.scale.stateScale.flatten()
            self.final_state_bound = self.final_state_bound / self.scale.stateScale.flatten()
            self.control_bound = self.control_bound / self.scale.controlScale.flatten()
            self.t0_bound = self.t0_bound / self.scale.sigmaScale
            self.tf_bound = self.tf_bound / self.scale.sigmaScale
            if self.guess_x0 is not None:
                self.guess_x0 = self.guess_x0 / self.scale.stateScale.flatten()
                self.guess_xf = self.guess_xf / self.scale.stateScale.flatten()
                self.guess_u0 = self.guess_u0 / self.scale.controlScale.flatten()
                self.guess_uf = self.guess_uf / self.scale.controlScale.flatten()
                self.guess_t0 = self.guess_t0 / self.scale.sigmaScale
                self.guess_tf = self.guess_tf / self.scale.sigmaScale
            # if self.guess_x is not None:
            #     self.guess_x = self.guess_x / self.scale.stateScale.flatten()
            # if self.guess_u is not None:
            #     self.guess_u = self.guess_u / self.scale.controlScale.flatten()
            # if self.guess_sigma is not None:
            #     self.guess_sigma = self.guess_sigma / self.scale.sigmaScale
        self.dimensionFlag = False

    def getAverageSigma(self):
        return (np.mean(self.tf_bound) - np.mean(self.t0_bound)) / 2  # (tf-t0)/2

    def getDimension(self):
        """ 获得xdim和udim
        确保bound数据是2×xdim和2×udim的 """
        assert self.init_state_bound.shape[0] == 2
        assert self.control_bound.shape[0] == 2
        return self.init_state_bound.shape[1], self.control_bound.shape[1]

    def isFreeTime(self):
        if self.t0_bound[0] == self.t0_bound[1] and self.tf_bound[0] == self.tf_bound[1]:
            return False
        else:
            return True

    def checkVadility(self):
        for key, value in self.__dict__.items():
            assert value is not None


class PhaseData:
    def __init__(self):
        """ data of a single phase, either cvxpy variables or constant values """
        self.xtao = None
        self.utao = None
        self.state = None
        self.control = None
        self.sigma = None
        self.trState = None

        self.nuDynamics = None  # dynamics constraints
        self.nuPath = None
        self.nuBoundary = None

    def __copy__(self):
        return PhaseData()

    def getTime(self):
        xt = (self.xtao + 1) * self.sigma
        ut = (self.utao + 1) * self.sigma
        return xt, ut

    def setupDVs(self, manager, freeTime, sigma=None, nPath: int = 0, nBoundary: int = 0):
        self.xtao = manager.mesh.xtao
        self.utao = manager.mesh.utao
        self.state = cvx.Variable((manager.nx, manager.xdim))
        self.control = cvx.Variable((manager.nu, manager.udim))
        if freeTime:
            self.sigma = cvx.Variable(nonneg=True)
        else:
            self.sigma = sigma

        self.nuDynamics = cvx.Variable((manager.nx - 1, manager.xdim), nonneg=True)
        if nPath:
            self.nuPath = cvx.Variable((manager.ncp, nPath), nonneg=True)
        if nBoundary:
            self.nuBoundary = cvx.Variable((nBoundary,), nonneg=True)

    def getPhaseValues(self):
        pd = PhaseData()
        pd.xtao = self.xtao
        pd.utao = self.utao
        pd.state = self.state.value
        pd.control = self.control.value
        pd.nuDynamics = self.nuDynamics.value
        if isinstance(self.sigma, cvx.expressions.variable.Variable):  # free duration
            pd.sigma = self.sigma.value
        else:
            pd.sigma = self.sigma  # fixed duration
        return pd

    def getDimensionlizedData(self, phaseinfo):
        """ data of a single phase, either cvxpy variables or constant values """
        xt = (self.xtao + 1) * self.sigma * phaseinfo.scale.sigmaScale
        ut = (self.utao + 1) * self.sigma * phaseinfo.scale.sigmaScale
        state = self.state * phaseinfo.scale.stateScale.flatten()
        control = self.control * phaseinfo.scale.controlScale.flatten()
        return xt, state, ut, control


class ParamStruct:
    """
    参数包含：参考轨迹，近似矩阵，信赖域
    params includes reference trajectory, approximation matrices and trust region.
    """

    def __init__(self):
        self.ncp = 0  # collocation number

        # params for integration
        self.integralWeight4State = None
        self.integralWeight4Control = None

        # 伪谱法需要的矩阵；for pseudo-spectral methods
        self.matInt = None
        self.PIMIndex = None  # starting point index in the integral form
        self.PIMedIndex = None  # integrated states index in the integral form

        # 近似矩阵；approximation matrices
        self.A_tilde = None
        self.B_tilde = None
        self.B2_tilde = None
        self.F_tilde = None
        self.R_tilde = None

        # 参考轨迹；reference trajectory
        self.refTraj = PhaseData()

        self.tr_state = None
        self.tr_control = None
        self.tr_sigma = None


class Linkage:
    """ Linkage condition between phases """

    def __init__(self, left, right, index, bound):
        self.left = left  # left phase index
        self.right = right  # right phase index
        self.index = np.array(index).astype(int)  # state index
        self.diff = np.array(bound).astype(float)  # limit
        assert len(index) == self.diff.shape[1]


class Result:
    def __init__(self):
        self.cvxTime = 0.
        self.maxError = 0.
        self.stepNum = 0
        self.maxRungeError = 0.
        self.objective = 0.
        self.solution = None  # list of phaseData
        self.solutionDimension = None
        self.solutionIntegrated = None
        self.meshHistory = []
        self.errorHistory = []

    def addcvxtime(self, ms):
        self.cvxTime += ms
