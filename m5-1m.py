import numpy as np
import matplotlib.pyplot as plt
import pickle
import utils
from OFTPLHints import OFTPLHints
from SCore import SCore
from RankedBandits import RankedBandits

# loading the data
with open("data.pickle", 'rb') as file:
    data = pickle.load(file)

# retrieving information from data
N = data.N
k = 10
time_horizon = data.time_horizon
ratedMovies = data.ratedMovies
perfectHints = data.perfectHints
randomHints=  data.randomHints
G = 1

# declaring the policies
oftplPerfectHints = OFTPLHints(N, k, 11)
oftplRandomHints = OFTPLHints(N, k, 11)
s_core = SCore(N, k, G, time_horizon)
rankedBandits = RankedBandits(N, k, time_horizon)

perfectHints_augmented_regret = []
randomHints_augmented_regret = []
s_core_augmented_regret = []
rankedBandits_augmented_regret = []

# running the policies and plotting augmented regret
for t in range(1, time_horizon + 1):
    perfectHint = perfectHints[t - 1]
    randomHint = randomHints[t - 1]
    reward = ratedMovies[t - 1]

    # get predictions
    perfectHints_prediction = oftplPerfectHints.getKSet(perfectHint)
    randomHints_prediction = oftplRandomHints.getKSet(randomHint)
    s_core_prediction = s_core.getKSet()
    rankedBandits_prediction = rankedBandits.getKSet()

    # feed the reward sets
    oftplPerfectHints.feedReward(reward, perfectHint)
    oftplRandomHints.feedReward(reward, randomHint)
    s_core.feedReward(reward)
    rankedBandits.feedReward(reward)

    # update augmented regret
    perfectHints_current_regret = (k/N) - utils.setIntersection(perfectHints_prediction, reward)
    randomHints_current_regret = (k/N) - utils.setIntersection(randomHints_prediction, reward)
    s_core_current_regret = (k/N) - utils.setIntersection(s_core_prediction, reward)
    rankedBandits_current_regret = (k/N) - utils.setIntersection(rankedBandits_prediction, reward)

    perfectHints_augmented_regret.append(perfectHints_current_regret)
    randomHints_augmented_regret.append(randomHints_current_regret)
    s_core_augmented_regret.append(s_core_current_regret)
    rankedBandits_augmented_regret.append(rankedBandits_current_regret)

# plotting
plt.xlabel("Time Horizon")
plt.ylabel("Augmented Regret")
plt.plot(range(1, time_horizon + 1), np.cumsum(np.array(perfectHints_augmented_regret)), label="perfectHints")
plt.plot(range(1, time_horizon + 1), np.cumsum(np.array(randomHints_augmented_regret)), label="randomHints")
plt.plot(range(1, time_horizon + 1), np.cumsum(np.array(s_core_augmented_regret)), label="s-core")
plt.plot(range(1, time_horizon + 1), np.cumsum(np.array(rankedBandits_augmented_regret)), label="rankedBandits")
plt.legend()
plt.savefig("plots/withRankedBandits(k = 10).png")
