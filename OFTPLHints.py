import math
import numpy as np
from Policy import Policy
import utils

class OFTPLHints(Policy):
    """
    This class implements the optimistic follow the perturbed
    leader policy for the subset selection problem, which utilizes
    hints. Refer to the paper to understand the parameters in the
    constructor.
    """
    def __init__(self, N, k, C, seed=1):
        """
        This constructor initializes the following properties of the policy.

        1. ``p``, the inclusion probability vector.
        2. ``eta``, the perturbation parameter.
        3. ``gamma``, the perturbation vector.
        4. ``cumulativeGradient``, the sum of all gradients observed till a given iteration.
        5. ``l1errors``, the sum of the squares of the L1 norms between gradients and hints.
        6. ``scale``, the scale factor used to calculate ``eta`` in each iteration.

        :param N: Number of elements in the ground set.
        :param k: Size of the subsets to be selected in each iteration.
        :param C: A parameter. Should satisfy ``C >= 11`` and ``2*C <= N``.

        :raises ValueError: if the conditions ``C >= 11`` and ``2*C <= N`` are not satisfied.
        """
        super().__init__(N, k)
        if C < 11 or 2*C > self.N:
            raise ValueError("The conditions C >= 11 and 2*C <= N must be satisfied!")
        self.C = C
        self.p = np.zeros(shape=self.N)
        self.eta = 0
        self.gamma = np.random.default_rng(seed=seed).normal(0, 1, size=self.N)
        self.cumulativeGradient = np.zeros(shape=self.N)
        self.l1errors = 0
        self.scale = (1.3/math.sqrt(self.C))*math.pow(1/math.log((self.N*math.e)/self.C), 1/4)

    def getKSet(self, hint):
        self.eta = self.scale*self.l1errors
        self.p = utils.linearOptimize(self.cumulativeGradient + hint + self.eta*self.gamma, self.N, self.k)
        return utils.MadowSample(self.p, self.N, self.k)

    def feedReward(self, reward, hint):
        self.T = self.T + 1

        # among all vertices of the 1-core, pick the one l1-closest to hint
        # here we use the fact that vertices of the 1-core are standard basis
        # vectors corresponding to elements in reward
        l1dist = None
        closestElement = None
        l1hint = np.linalg.norm(hint, ord=1)
        g = np.zeros(shape=self.N, dtype=np.int_)
        for element in reward:
            # if hint contains element, then minimum l1-distance is l1dist - 1
            # and the vertex l1-closest to hint is the standard basis vector
            # corresponding to element
            closestElement = element
            if hint[element - 1]:
                l1dist = l1hint - 1
                break
            else:
                l1dist = l1hint + 1

        assert(l1dist != None)
        assert(closestElement != None and 1 <= closestElement <= self.N)

        # set g to be the standard basis vector corresponding to
        # closestElement
        g[closestElement - 1] = 1

        self.l1errors = math.sqrt(self.l1errors**2 + l1dist)
        self.cumulativeGradient = self.cumulativeGradient + g


