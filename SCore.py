import math
import numpy as np
from Policy import Policy
import utils

class SCore(Policy):
    """
    This class implements the SCore policy for online subset
    selection. This policy does not use hints.
    """
    def __init__(self, N, k, G, time_horizon):
        """
        This constructor initializes the following properties of the policy.
        1. ``p``, the inclusion probability vector.
        2. ``cumulativeGradient``, sum of all gradients observed till the current iteration.
        3. ``eta``, learning rate, which is required for the optimization step.

        :param N: Number of elements in the ground set.
        :param k: Size of the subsets to be selected in each iteration.
        :param G: Upper bound on the L2 norms of the vectors in the 1-cores of reward functions.
        :param time_horizon: The time horizon, i.e the total number of iterations.
        """
        super().__init__(N, k)
        self.G = G
        self.time_horizon = time_horizon
        self.p = np.ones(shape=self.N) * (k/N)
        self.cumulativeGradient = np.zeros(shape=self.N)
        self.eta = math.sqrt((k*math.log(N/k))/(2*self.G*self.G*self.time_horizon))

    def getKSet(self, hint=None):
        return utils.MadowSample(self.p, self.N, self.k)

    def feedReward(self, reward, hint=None):
        self.T = self.T + 1
        element = reward[0]

        # pick a vector in 1-core
        # vertices of 1-core are standard basis vectors
        # corresponding to elements in reward
        g = np.zeros(shape=self.N, dtype=np.int_)
        g[element - 1] = 1
        self.cumulativeGradient = self.cumulativeGradient + g

        # compute next probability vector
        self.p = utils.ftrlOptimize(self.cumulativeGradient, self.N, self.k, self.eta)