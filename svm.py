import numpy as np
import pandas as pd
# import statsmodels.ap as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.utils import shuffle


# def remove_correlated_features(X):


# def remove_less_significant_features(X, Y):


def compute_cost(W, X, Y):
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0
    hinge_loss = reg_strength * (np.sum(distances) / N)

    cost = 1/2 * np.dot(W, W) + hinge_loss
    return cost


def calculate_cost_gradient(W, X_batch, Y_batch):
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])

    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))

    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (reg_strength * Y_batch[ind] * X_batch[ind])
        dw += di
    dw = dw/len(Y_batch)


def sgd(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01
    for epoch in range(1, max_epochs):
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)

        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            print("Epoch is: {} and Cost is: {}".format(epoch, cost))
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1
    return weights


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

    print("training started...")
    W = sgd(X_train.to_numpy(), y_train.to_numpy())
    print("training finished.")
    print("weights are: {}".format(W))

    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(W, X_test.to_numpy()[i]))
        y_test_predicted = np.append(y_test_predicted, yp)

    print("accuracy on test dataset: {}".format(
        accuracy_score(y_test.to_numpy(), y_test_predicted)))
    print("recall on test dataset: {}".format(
        recall_score(y_test.to_numpy(), y_test_predicted)))
    print("precision on test dataset: {}".format(
        recall_score(y_test.to_numpy(), y_test_predicted)))


reg_strength = 10000
learning_rate = 0.000001
init()
