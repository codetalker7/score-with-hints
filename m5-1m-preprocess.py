import numpy as np
import pandas as pd
import pickle
from utils import DataLoader

ratings_file = "data/ml-1m/ratings.dat"
movies_file = "data/ml-1m/movies.dat"
users_file = "data/ml-1m/users.dat"

# loading the data as dataframes
ratings = pd.read_csv(ratings_file, sep="::", names=["UserID", "MovieID", "Rating", "TimeStamp"], encoding='ascii')
movies = pd.read_csv(movies_file, sep="::", names=["MovieID", "Title", "Genres"], encoding='ISO-8859-1')
users = pd.read_csv(users_file, sep="::", names=["UserID", "Gender", "Age", "Occupation", "Zip-code"], encoding='ascii')

# getting number of movies and users
numMovies = movies["MovieID"].max()
numUsers = users["UserID"].max()

# generating lists of movies and hints
ratedMovies = []
perfectHints = []
randomHints = []
for user in range(1, numUsers + 1):
    userMovies = ratings[ratings["UserID"] == user]
    listOfMovies = list(userMovies["MovieID"].unique())
    assert(len(listOfMovies) >= 20)
    ratedMovies.append(listOfMovies)

    perfect_hint = np.zeros(shape=numMovies, dtype=np.int_)
    perfect_hint[listOfMovies[0] - 1] = 1
    perfectHints.append(perfect_hint)

    random_hint = np.zeros(shape=numMovies, dtype=np.int_)
    for i in range(1, numMovies + 1):
        probability = np.random.uniform()
        if i in listOfMovies:
            if probability < 0.3:
                random_hint[i - 1] = 1
        else:
            if probability < 0.1:
                random_hint[i - 1] = 1
    randomHints.append(random_hint)
    
# serializing the objects
data = DataLoader(numMovies, numUsers, ratedMovies, perfectHints, randomHints)
with open("data.pickle", 'wb') as file:
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

