import numpy as np
import pandas as pd
# import statsmodels.ap as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


# def remove_correlated_features(X):


# def remove_less_significant_features(X, Y):


# def compute_cost(W, X, Y):
# def calculate_cost_gradient(W, X_batch, Y_batch):
# def sgd(features, outputs):


def init():
    data = pd.read_csv('.data.csv')
    diagnosis_map = {'M': 1, 'B': -1}
    data['diagnosis'] = data['diagnosis'].map(diagnosis_map)
    data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

    Y = data.loc[:, 'diagnosis']
    X = data.iloc[:, 1:]

    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)

    X.insert(loc=len(X.columns), column='intercept', value=1)
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(
        X, Y, test_size=0.2, random_state=42)
