import math
import numpy as np
from Policy import Policy
import utils

class SCore(Policy):
    """
    This class implements the SCore policy for online subset
    selection. This policy does not use hints.

    Parameters:
    `p`: Inclusion probability vector
    `cumulativeGradient`: Sum of all gradients observed till the current iteration.
    """
    def __init__(self, N, k):
        super().__init__(N, k)
        self.p = np.ones(shape=self.N) * (k/N)
        self.cumulativeGradient = np.zeros(shape=self.N)

    def getKSet(self, hint):
        return utils.MadowSample(self.p, self.N, self.k)

    def feedReward(self, reward, hint):
        self.T = self.T + 1
        element = reward[0]

        # pick a vector in 1-core
        # vertices of 1-core are standard basis vectors
        # corresponding to elements in reward
        g = np.zeros(shape=self.N, dtype=np.int_)
        g[element - 1] = 1
        self.cumulativeGradient = self.cumulativeGradient + g

        # compute next probability vector
        self.p = utils.ftrlOptimize(self.cumulativeGradient)