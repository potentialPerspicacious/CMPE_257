import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def sigmoid(z):
    return 1 / 1 + np.exp(-z)


def initialize_weights(l_units):
    np.random.seed(4)
    w1 = np.random.uniform(-0.05, 0.05, (l_units, 4))
    b1 = np.ones(shape=(l_units, 1))
    w2 = np.random.uniform(-0.05, 0.05, (3, l_units))
    b2 = np.ones(shape=(3, 1))
    parameter = {"w1": w1,
                 "b1": b1,
                 "w2": w2,
                 "b2": b2}
    print("weights of first layer", w1.shape)
    # print("Transpose of above vector", w1.T.shape)
    # print("bias of layer 1", b1.shape)
    # print("weights of layer 2", w2.shape)
    # print("bias of layer 2", b2.shape)

    return parameter


def propogation(X, Y, parameter):
    m = X.shape[1]
    w1 = parameter['w1']
    b1 = parameter['b1']
    w2 = parameter['w2']
    b2 = parameter['b2']
    #print("w1 ka shape", w1.shape)
    #print("XXX shape", X.shape)

    # fwdpropogation
    A1 = sigmoid((w1 @ X) + b1)
    A2 = sigmoid((w2 @ A1) + b2)
    # print("A1 shape:", A1.shape)
    #print("A2 shape:", A2.shape)
    #print("A2 shape@@@", A2.shape[1])

    # cost = (-1. / m) * np.sum(Y * np.log(A2) + (1 - Y) * (np.log(1 - A2)))
    log = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = -np.sum(log) / m

    # Backpropogation
    n = Y.shape[1]
    # print("x.shape", n)
    # print("what the hell is this", '\n', Y.shape)
    dw2 = (1. / n) * ((A2 - Y) @ A1.T)
    db2 = (1. / n) * np.sum(A2 - Y)
    z = np.multiply((w2.T @ (A2 - Y)), 1 - np.power(A1, 2))
    dw1 = (1. / n) * z @ X.T
    db1 = (1. / n) * np.sum(z)

    cost = np.squeeze(cost)

    grad_desc = {'dw1': dw1, 'db1': db1,
                 'dw2': dw2, 'db2': db2}

    return grad_desc, cost


def update_parameters(parameter, grad_desc, learning_rate=0.01):
    dw1 = grad_desc['dw1']
    db1 = grad_desc['db1']
    dw2 = grad_desc['dw2']
    db2 = grad_desc['db2']
    w1 = parameter['w1']
    b1 = parameter['b1']
    w2 = parameter['w2']
    b2 = parameter['b2']

    # grad_desc, cost = propogation(w1, b1, w2, b2, x, y, parameter)
    # for a in range(0, len(dw1)):
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    # for q in range(0, len(dw2)):
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2

    parameter = {'w1': w1, 'b1': b1,
                 'w2': w2, 'b2': b2}
    grad_desc = {'dw1': dw1, 'db1': db1,
                 'dw2': dw2, 'db2': db2}

    return parameter, grad_desc


def predict(x_test, parameter):
    m = y.shape[1]
    y_pred = np.zeros((1, m))
    w1 = parameter['w1']
    w2 = parameter['w2']
    b1 = parameter['b1']
    b2 = parameter['b2']


    A1 = sigmoid(np.dot(w1, x_test) + b1)
    A2 = sigmoid(np.dot(w2, A1) + b2)


    for i in range(A2.shape[1]):
        y_pred[0, i] = 1 if A2[0, i] > 0.9 else 0

    assert (y_pred.shape == (1, m))

    return y_pred


def nn_model(X_train, Y_train, X_test, Y_test, learning_rate=0.01, epochs=10000, print_cost=False):
    np.random.seed(3)
    # print("somewhere inside", Y_train.shape)
    parameter = initialize_weights(6)
    w1 = parameter['w1']
    w2 = parameter['w2']
    b1 = parameter['b1']
    b2 = parameter['b2']
    print("whats here", '\n', parameter)

    for i in range(1, epochs):
        print("w1:", parameter)
        grad_desc, costs = propogation(X_train, Y_train, parameter)
        # print("w1", w1)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, costs))

        parameter = update_parameters(parameter, grad_desc)



    Y_prediction_test = predict(w2, b2, X_test)
    Y_prediction_train = predict(w2, b2, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {'costs': costs,
         'Y_prediction_test': Y_prediction_test,
         'Y_prediction_train': Y_prediction_train,
         'w': w2,
         'b': b2,
         'learning_rate': learning_rate,
         'epochs': epochs}

    return d


load_data = pd.read_csv('iris.csv')
# print(load_data)

load_data.loc[load_data['variety'] == 'Virginica', 'variety'] = 2
load_data.loc[load_data['variety'] == 'Versicolor', 'variety'] = 1
load_data.loc[load_data['variety'] == 'Setosa', 'variety'] = 0
x = load_data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = load_data[['variety']]
he = OneHotEncoder(sparse=False)
y = he.fit_transform(y)

for i in range(y.shape[0]):
    for j in range(y.shape[1]):
        if y[i][j] == 0:
            y[i][j] = 0.1
        else:
            y[i][j] = 0.9

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

nn_model(x_train, y_train, x_test, y_test, epochs=10000, learning_rate=0.01, print_cost=True)
