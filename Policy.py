from abc import ABC, abstractmethod

class Policy(ABC):
    """
    A ``Policy`` object represents an online policy for the
    subset selection problem.
    """
    def __init__(self, N, k):
        """
        Sets the number of elements in the ground set, and the parameter
        ``k``, denoting the size of subsets which need to be picked in each step.
        Also initializes ``T`` to 0, which is the the number of steps for 
        which the policy has been trained.

        :param N: Number of elements in the ground set.
        :param k: Size of the subset to be selected in each iteration.
        """
        self.N = N
        self.k = k
        self.T = 0

    @abstractmethod
    def getKSet(self, hint):
        """
        Get a ``k`` sized subset in a particular iteration using the
        ``hint`` if necessary.

        :param hint: Characteristic vector of the hint set. Policies which don't use hints can simply ignore this argument.

        :returns: A ``k``-sized list of elements picked by the policy.
        """
        pass

    @abstractmethod
    def feedReward(self, reward, hint):
        """
        Feed the observed reward to the policy.

        :param reward: List of elements in the reward set.
        :param hint: Characteristic vector of the hint set.
        """
        pass

    