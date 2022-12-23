from abc import ABC, abstractmethod

class Policy(ABC):
    """
    A `Policy` object represents an online policy for the
    subset selection problem.
    """
    def __init__(self, N, k):
        """
        Arguments:
        `N`: Number of elements in the set.
        `k`: Size of the subset to be selected in each iteration.
        """
        self.N = N
        self.k = k

    @abstractmethod
    def getKSet(hint):
        """
        Get a `k` sized subset in a particular iteration using the
        `hint` if necessary.

        Arguments:
        `hint`: Characteristic vector of the hint set.

        Return value:
        Characteristic vector of the set which is picked by the policy.
        """
        pass

    @abstractmethod
    def feedReward(reward):
        """
        Feed the observed reward to the policy.

        Arguments:
        `reward`: Characterstic vector of the reward set.
        """
        pass

    