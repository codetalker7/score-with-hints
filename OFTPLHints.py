import numpy as np
from Policy import Policy
import utils

class OFTPLHints(Policy):
    def __init__(self, N, k, C, seed):
        super().__init__(N, k)
        self.C = C
        self.p = np.ones((self.N, 1)) * (self.k/self.N)
        self.eta = 0
        self.gamma = np.random.default_rng(seed=seed).normal(0, 1, size=(self.N, 1))    
        self.cumulativeGradient = np.zeros(shape=(self.N, 1))

    def getInclusionProbabilities(self, hint):
        return utils.linearOptimize(self.cumulativeGradient + hint + self.eta*self.gamma)

    def getKSet(hint):
        pass

    def feedReward():
        pass

