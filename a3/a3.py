# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/
# Note that I have not provided many doctests for this one. I strongly
# recommend that you write your own for each function to ensure your
# implementation is correct.

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/p9wmkvbqt1xr6lc/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    pass
    token = [[tokenize_string(g)] for g in movies.genres]
    movies = movies.join(pd.DataFrame(token, columns=["tokens"]))
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    pass
    unique_term_list = list()

    for token in movies.tokens:
        token1 = defaultdict(int)
        for token2 in token:
            token1[token2] = token1[token2] + 1
        unique_term_list.append(token1)
    # print(unique_term_list)
    
    features = defaultdict(int)
    for token in movies.tokens:
        for set1 in set(token):
            features[set1] = features[set1] + 1
    # print(features) 
    feature_list = sorted(features)
    vocab = dict((i, feature_list.index(i)) for i in feature_list)
    #print(vocab)
    
    # CSR matrix
    CSR1 = []
    for i in range(len(movies)):
        unique = unique_term_list[i]
        max1 = unique[max(unique, key = unique.get)]
        row = []
        col = []
        stats = []
        shapes = (1, len(vocab))
        for j in unique:
            row.append(0)
            col.append(vocab[j])
            tfidf = unique[j] / max1 * math.log10(len(movies) / features[j])
            stats.append(tfidf)
        CSR1.append(csr_matrix((stats, (row, col)), shape = shapes))

    movies = movies.assign(features=pd.Series(CSR1,movies.index))
    result = tuple((movies, vocab))
    return result


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      A float. The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    pass
    A = (np.dot(a, b.T).toarray()[0][0])
    B = (np.linalg.norm(a.toarray()) * np.linalg.norm(b.toarray()))
    return float(A/B)


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    pass
    result = list()
    for user_id, movie_id in (ratings_test.iterrows()):
        movie_id = ratings_test.movieId
        user_id = ratings_test.userId    
    for user, movie in zip(user_id,movie_id):
        user_ratings = ratings_train[ratings_train.userId == user]
        matched_movie = movies[movies.movieId == movie].iloc[0]
        B = matched_movie.features
        user_ratings = user_ratings.join(movies.set_index('movieId'),on = 'movieId')
        weighted = 0
        cos = 0
        rating1 = user_ratings.rating
        features = user_ratings.features
        for rating_1, A in zip(rating1,features):
            COSINE_SIM1 = cosine_sim(A,B)
            if COSINE_SIM1 > 0:
                weighted += COSINE_SIM1*rating_1
                cos+=COSINE_SIM1
                zero = True
        if cos > 0:
            result.append(float(weighted/cos))
        elif cos == 0:
            result.append(float(user_ratings.rating.mean()))
    result = np.array(result)        
    return result



def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
