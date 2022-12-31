import numpy as np
import math
from Policy import Policy

class RankedBandits(Policy):
    """
    Class representing the Ranked Bandits Algorithm for learning rankings
    with multi-armed bandits. This algorithm does not utilize hints.
    """
    def __init__(self, N, k, time_horizon):
        """
        This constructor initializes the following properties of the policy.

        1. ``mabs``, a list of ``k`` multi-armed bandits trained using the EXP3 algorithm.
        2. ``arms``, a ``numpy.array`` of arms selected in an iteration by each of the multi-armed bandits.
        3. ``predictedSet``, a ``k``-sized list of elements picked by the policy.

        :param N: Number of elements in the ground set.
        :param k: Size of the subsets to be selected in each iteration.
        :param time_horizon: The time horizon, i.e the total number of iterations.
        """
        super().__init__(N, k)
        self.time_horizon = time_horizon
        # initialize k multi-armed bandits, each with N arms
        self.mabs = [MultiArmedBandit(self.N, self.time_horizon) for i in range(self.k)]
        self.arms = np.zeros(shape=self.k, dtype=np.int_)
        self.predictedSet = []

    def getKSet(self, hint=None):
        selectedDocuments = np.zeros(shape=self.N, dtype=np.int_)
        self.predictedSet = []
        for i in range(self.k): # determining document for each rank
            # get the predicted arm for the ith mab
            self.arms[i] = self.mabs[i].selectArm()

            if self.arms[i] in self.predictedSet:
                # select arbitrary unselected document
                unselectedDocumentIndex = np.argmin(selectedDocuments)
                selectedDocuments[unselectedDocumentIndex] = 1
                self.predictedSet.append(unselectedDocumentIndex + 1)
            else:
                # select the predicted arm
                selectedDocuments[self.arms[i] - 1] = 1
                self.predictedSet.append(self.arms[i])

        return self.predictedSet

    def feedReward(self, reward, hint=None):
        self.T = self.T + 1

        userClicked = False
        for i in range(self.k):
            ith_prediction = self.predictedSet[i]
            if not(userClicked) and ith_prediction in reward and ith_prediction == self.arms[i]:
                userClicked = True
                mab_reward = 1
            else:
                mab_reward = 0
            self.mabs[i].update(self.arms[i], mab_reward)

class MultiArmedBandit:
    """
    Multi-Armed Bandit trained using the EXP3 algorithm.
    """
    def __init__(self, N, time_horizon):
        """
        This constructor initializes a multi-armed bandit.

        :param N: Number of elements in the choice set.
        :param time_horizon: The time horizon, i.e the number of iterations.
        """
        self.N = N
        self.time_horizon = time_horizon
        self.gamma = min(1, math.sqrt((self.N*math.log(self.N))/((math.e - 1)*self.time_horizon)))
        self.weights = np.ones(shape=self.N)
        self.p = np.ones(shape=self.N)/N

    def selectArm(self):
        """
        Make the next prediction.
        """
        self.p = ((1 - self.gamma)/np.sum(self.weights))*self.weights + self.gamma/self.N
        return np.random.choice(self.N, p=self.p) + 1

    def update(self, arm, reward):
        """
        Do the EXP3 update step.

        :param arm: Most recent prediction made by the multi-armed bandit.
        :param reward: The reward for the prediction.
        """
        self.weights[arm - 1] *= math.exp((reward*self.gamma)/(self.p[arm - 1]*self.N))

