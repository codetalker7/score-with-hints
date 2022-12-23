def linearOptimize(cost):
    """
    Performs linear optimization over the set of inclusion
    probability vectors, i.e the set of all vectors p such
    that 0 <= p_i <= 1 for each i, and sum_{i} p_i = k.

    Arguments: 
    cost -- the cost vector

    Return value:
    An inclusion probability vector p such that p*w is maximum. 
    Here p*w is the inner product of p and w.
    """
    pass

def MadowSample(p, N, k):
    """
    Samples k elements from the set [N] using Madow's sampling
    algorithm with inclusion probabilities given by the vector
    p.

    Arguments: 
    p -- inclusion probability vector
    N -- size of the set to sample from
    k -- number of elements to sample

    Return value:
    Characteristic vector of the subset of [N] which is sampled.
    """
    pass