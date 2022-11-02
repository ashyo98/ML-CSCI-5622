import os
import pickle
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split

current_folder = os.path.dirname(os.path.abspath(__file__))

class HousePrices(object):
    def __init__(self):
        rawdata = pd.read_csv('data/kc_house_data.csv').drop(['id', 'date', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'], axis=1).to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(rawdata[:,1:], rawdata[:,0], test_size=0.2, random_state=5622)

class Digits(object):

    def __init__(self):
        loaded = np.load(os.path.join(current_folder, "mnist.npz"))
        self.images = images = loaded["images"].reshape(-1, 28 * 28)
        self.labels = labels = loaded["labels"]
        train_size = 1000
        valid_size = 500
        test_size = 500
        self.X_train, self.y_train = images[:train_size], labels[:train_size]
        self.X_valid, self.y_valid = images[train_size: train_size + valid_size], labels[
                                                                                  train_size: train_size + valid_size]
        self.X_test, self.y_test = (images[train_size + valid_size:train_size + valid_size + test_size],
                                    labels[train_size + valid_size: train_size + valid_size + test_size])


class BinaryDigits:
    """
    Class to store MNIST data for images of 9 and 8 only
    """

    def __init__(self):
        loaded = np.load(os.path.join(current_folder, "mnist.npz"))
        images = loaded["images"].reshape(-1, 28 * 28)
        labels = loaded["labels"]
        labels = labels % 2
        train_size = 1000
        valid_size = 500
        test_size = 500

        self.X_train, self.y_train = images[:train_size], labels[:train_size]
        self.X_valid, self.y_valid = images[train_size: train_size + valid_size], labels[
                                                                                  train_size: train_size + valid_size]
        self.X_test, self.y_test = (images[train_size + valid_size:train_size + valid_size + test_size],
                                    labels[train_size + valid_size: train_size + valid_size + test_size])


class IMDB:
    """
    Class to store IMDB dataset
    """

    def __init__(self):
        with open(os.path.join(current_folder, "movie_review_data.json")) as f:
            self.data = data = json.load(f)
        X = [d['text'] for d in data['data']]
        y = [d['label'] for d in data['data']]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, shuffle=True,
                                                                                random_state=42)
