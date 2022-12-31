import math
import numpy as np

def linearOptimize(cost, N, k):
    """
    Performs linear optimization over the set of inclusion
    probability vectors, i.e the set of all vectors p such
    that 0 <= p_i <= 1 for each i, and sum_{i} p_i = k.


    :param cost: The cost vector.
    :param N: The dimension of the underlying space.
    :param k: Size of the subsets selected in each iteration.

    :returns: An inclusion probability vector ``p`` such that ``p*w`` is maximum. Here ``p*w`` is the inner product of ``p`` and ``w``.
    """
    p = np.zeros(shape=N, dtype=np.int_)

    # get the indices which sort the cost vector
    sorted_indices = np.argsort(cost)

    # put 1 in the k largest indices of p
    for i in range(N - k, N):
        p[sorted_indices[i]] = 1

    return p

def ftrlOptimize(cumulativeGradient, N, k, eta):
    """
    Performs the optimization step for the Follow the Regularized Leader
    (FTRL) framework with entropic regularizer over the set of inclusion
    probability vectors.

    :param cumulativeGradient: Total gradient seen until the previous iteration.
    :param N: Size of the ground set.
    :param k: Size of the subset to be picked.
    :param eta: Learning rate.

    :returns: An inclusion probability vector ``p`` such that ``p*cumulativeGradient - p*log(p)`` is maximized.
    """
    # sort cumulativeGradient in non-increasing order
    orderedVector = -np.sort(-cumulativeGradient)

    # finding i_star
    i_star = N
    tail_sum = 0
    while i_star >= 1:
        if (k - i_star)*math.exp(eta*orderedVector[i_star - 1]) >= tail_sum:
            break
        else:
            tail_sum = tail_sum + math.exp(eta*orderedVector[i_star - 1])
            i_star = i_star - 1

    # computing K
    if i_star == N:     # we will have k = N in this case
        return np.ones(shape=N)

    # assuming that i_star < N
    K = (k - i_star)/tail_sum
    p = np.zeros(shape=N)
    for i in range(1, N + 1):
        p[i - 1] = min(1, K*math.exp(eta*orderedVector[i - 1]))
    return p


def MadowSample(p, N, k):
    """
    Samples ``k`` elements from the set ``[N]`` using Madow's sampling
    algorithm with inclusion probabilities given by the vector
    ``p``.

    Arguments: 
    :param p: Inclusion probability vector
    :param N: Size of the set to sample from
    :paran k: Number of elements to sample

    :returns: List of elements of ``[N]`` which are sampled.
    """
    pi = np.cumsum(np.insert(p, 0, 0))
    U = np.random.uniform()
    S = []

    for i in range(0, k):
        for j in range(1, N + 1):
            if (pi[j - 1] <= U + i < pi[j]):
                S.append(j)

    return S

def setIntersection(A, B):
    """
    Determine whether ``A`` and ``B`` have non-empty intersection.

    :param A: The first set
    :param B: The second set

    :returns: ``1``, if ``A`` and ``B`` intersect, and ``0`` otherwise.
    """
    if (set(A) & set(B)):
        return 1
    else:
        return 0

class DataLoader:
    def __init__(self, N, time_horizon, ratedMovies, perfectHints, randomHints):
        self.N = N
        self.time_horizon = time_horizon
        self.ratedMovies = ratedMovies
        self.perfectHints = perfectHints
        self.randomHints = randomHints