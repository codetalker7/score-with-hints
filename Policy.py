from abc import ABC, abstractmethod

class Policy(ABC):
    """
    A `Policy` object represents an online policy for the
    subset selection problem.
    """
    def __init__(self, N, k):
        """
        Sets the number of elements in the ground set, and the parameter
        `k`, denoting the size of subsets which need to be picked in each step.
        `T` represents  the number of steps for which the policy has been trained.

        Arguments:
        `N`: Number of elements in the set.
        `k`: Size of the subset to be selected in each iteration.
        """
        self.N = N
        self.k = k
        self.T = 0

    @abstractmethod
    def getKSet(self, hint):
        """
        Get a `k` sized subset in a particular iteration using the
        `hint` if necessary.

        Arguments:
        `hint`: Characteristic vector of the hint set. Policies which
        don't use hints can simply ignore this argument.

        Return value:
        Characteristic vector of the set which is picked by the policy.
        """
        pass

    @abstractmethod
    def feedReward(self, reward, hint):
        """
        Feed the observed reward to the policy.

        Arguments:
        `reward`: List of elements in the reward set.
        `hint`: Characteristic vector of the hint set.
        """
        pass

    