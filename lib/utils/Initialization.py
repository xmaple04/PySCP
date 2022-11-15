"""
author: xmaple
date: 2021-10-16
"""
import numpy as np


class Initialization:
    def __init__(self, method='linear'):
        self.method = method

    def getInitialTrajectory(self, guess):
        guess_init_state = (self.init_state_min + self.init_state_max) / 2
        guess_final_state = (self.final_state_min + self.final_state_max) / 2

        nodes_num = len(nodes)
        collp_num = nodes_num - 1
        X = guess_init_state.reshape((-1, 1)) * (1 - nodes.reshape((1, -1))) / 2 + \
            guess_final_state.reshape((-1, 1)) * (1 + nodes.reshape((1, -1))) / 2
        U = np.zeros((self.udim, collp_num))

        U[0, :] = self.control_max[0] / 2
        U[2, :] = self.control_max[2] / 2

        if self.freetime:
            sigma = (self.sigma_max + self.sigma_min) / 2
        else:
            sigma = self.sigma
        return X, U, sigma
