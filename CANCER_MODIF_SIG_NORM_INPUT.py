
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import pandas as pd
import csv
import os
import time


#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder

#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data', header=None)

#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)


#Loading iris dataset from scikit learn datasets module
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer


dataset = load_breast_cancer()
number_of_classes = 2                      ## 3 for iris, wine, 2 for cancer
number_of_training_samples = 456          ## digits
#number_of_training_samples = 140            ##for wine

# Store features matrix in X
X = dataset.data

# Store target vector in
y = dataset.target





def plot_cost(cost, number_of_epochs):

#    print('shape of cost dictionary: ', cost.shape)
    plt.plot(list(cost.keys()), list(cost.values()))           ## to be wrapped in list() function

    plt.title("Test Error vs. #Epochs")

    plt.xlabel("#Epochs")
    plt.ylabel("Error")

#    plt.grid(True)
    plt.grid(False)

    if number_of_epochs <= 20:
        plt.xticks(list(range(0, number_of_epochs, 1)))           ## epochs on the x-axis  ## don't know how to start from EPOCH 1
    else:
        plt.xticks(list(range(0, number_of_epochs, 10)))

    plt.show()


def plot_accuracy(accu, number_of_epochs):

#    print('shape of cost dictionary: ', cost.shape)
    plt.plot(list(accu.keys()), list(accu.values()))           ## to be wrapped in list() function

    plt.title("Accuracy vs. #Epochs")

    plt.xlabel("#Epochs")
    plt.ylabel("Accuracy (%)")

#    plt.grid(True)
    plt.grid(False)


    if number_of_epochs <= 25:
        plt.xticks(list(range(0, number_of_epochs, 1)))  ## epochs on the x-axis  ## don't know how to start from EPOCH 1
    elif 25 < number_of_epochs <= 150:
        plt.xticks(list(range(0, number_of_epochs, 10)))  ## epochs on the x-axis  ## don't know how to start from EPOCH 1
    elif 150 < number_of_epochs <= 450:
        plt.xticks(list(range(0, number_of_epochs, 40)))  ## epochs on the x-axis  ## don't know how to start from EPOCH 1
    else:
        plt.xticks(list(range(0, number_of_epochs, 400)))

#    plt.yticks(list(range(0, 100, 5)))
    plt.show()




def sqish(z):
    s = np.zeros((z.shape[0], z.shape[1]))
    for s_x in range(z.shape[0]):
        for s_y in range(z.shape[1]):
            if z[s_x][s_y] > 0:
                s[s_x][s_y] = z[s_x][s_y] + ((z[s_x][s_y] * z[s_x][s_y])  /  32)
            elif -2 <= z[s_x][s_y] < 0:
                s[s_x][s_y] = z[s_x][s_y] + ((z[s_x][s_y] * z[s_x][s_y])  /  2)
            else:
                s[s_x][s_y] = 0
    return s


def deriv_sqish(z):
    s = np.zeros((z.shape[0], z.shape[1]))
    for s_x in range(z.shape[0]):
        for s_y in range(z.shape[1]):
            if z[s_x][s_y] > 0:
                s[s_x][s_y] = 1 + ((z[s_x][s_y]) / 16)
            elif -2 <= z[s_x][s_y] < 0:
                s[s_x][s_y] = 1 + (z[s_x][s_y])
            else:
                s[s_x][s_y] = 0
    return s



def sigmoid(z):
    s = np.zeros((z.shape[0], z.shape[1]))
    for s_x in range(z.shape[0]):
        for s_y in range(z.shape[1]):
            if z[s_x][s_y] < -2:
                s[s_x][s_y] = 0
            elif -2 <= z[s_x][s_y] < 0:
                s[s_x][s_y] = 0.5 + (0.5 * (z[s_x][s_y] + (1/4 * z[s_x][s_y] * z[s_x][s_y])))
            elif 0 <= z[s_x][s_y] <= 2:
                s[s_x][s_y] = 0.5 + (0.5 * (z[s_x][s_y] - (1/4 * z[s_x][s_y] * z[s_x][s_y])))
            else:
                s[s_x][s_y] = 1
    return s


def deriv_sigmoid(z):
    s = np.zeros((z.shape[0], z.shape[1]))
    for s_x in range(z.shape[0]):
        for s_y in range(z.shape[1]):
            if 0 <= z[s_x][s_y] <= 2:
                s[s_x][s_y] = (2 - (z[s_x][s_y])) / 4
            elif -2 <= z[s_x][s_y] < 0:
                s[s_x][s_y] = (2 + (z[s_x][s_y])) / 4
            else:
                s[s_x][s_y] = 0
    return s


def x3(z):
#    return np.maximum(0, z)
    s = np.zeros((z.shape[0], z.shape[1]))                                   ###if you don't do this --- s not defined
    for s_x in range(z.shape[0]):
        for s_y in range(z.shape[1]):
            if z[s_x][s_y] < -2:
                s[s_x][s_y] = 1
            elif z[s_x][s_y] > 2:
                s[s_x][s_y] = 1
            else:
                s[s_x][s_y] = (z[s_x][s_y] * z[s_x][s_y] * z[s_x][s_y])
    return s



def x3_derivative(z):
    d = np.zeros((z.shape[0], z.shape[1]))
    for d_x in range(z.shape[0]):
        for d_y in range(z.shape[1]):
            if -2 <= z[d_x][d_y] <= 2:                 ###  BIGGEST ISSUE --- if you do >= 0 --- NO CONVERGENCE
                d[d_x][d_y] = 3 * z[d_x][d_y] * z[d_x][d_y]
            else:
                d[d_x][d_y] = 0
    return d


def relu(z):
#    return np.maximum(0, z)
    s = z                                   ###if you don't do this --- s not defined
    for s_x in range(z.shape[0]):
        for s_y in range(z.shape[1]):
            if z[s_x][s_y] <= 0:
                s[s_x][s_y] = 0
            else:
                s[s_x][s_y] = z[s_x][s_y]
# #   s = 1. / (1. + np.exp(-steepness_factor * z + 4.5))            # s auto. initialized   ## broadcasting being done
    return s


def relu_derivative(z):
    d = z
    for d_x in range(z.shape[0]):
        for d_y in range(z.shape[1]):
            if z[d_x][d_y] > 0:                 ###  BIGGEST ISSUE --- if you do >= 0 --- NO CONVERGENCE
                d[d_x][d_y] = 1
            else:
                d[d_x][d_y] = 0
#    s = 1. / (1. + np.exp(-steepness_factor * z + 4.5))            # s auto. initialized   ## broadcasting being done
    return d



def exact_accuracy(predicted_output, actual_output):

    correct = 0
    wrong = 0

#    print('shape of pred: ', predicted_output.shape)                ## (10000, )
#    print('shape of actual: ', actual_output.shape)                 ## (10000, )

    for all_examples in range(actual_output.shape[0]):
            if predicted_output[all_examples] == actual_output[all_examples]:
                correct = correct + 1
            else:
                wrong = wrong + 1

    exact_acc = (correct / (correct + wrong)) * 100

    return exact_acc





def compute_loss(Y, Y_hat):                                    ## Y and Y_hat .. same shape... (10, 1500), (10, 297)
    L_sum = np.sum(np.multiply((Y_hat - Y), (Y_hat - Y)))
#    L_sum = np.sum(np.abs(Y_hat - Y))
#    L_sum = np.sum(Y_hat - Y)          ## WRONG
    m = Y.shape[1]
    L = (1. / m) * L_sum                                   ## -ve sign does not impact results
#    L = -(1. / m) * L_sum                                 ## cross entropy and MSE...no impact

    return L

##[x,y]




def derivative_mae(y_pred, y_label):
    mae = np.zeros((y_pred.shape[0], y_pred.shape[1]))
    for pred_x in range(y_label.shape[0]):
        for pred_y in range(y_label.shape[1]):
            if (y_pred[pred_x][pred_y]  >  y_label[pred_x][pred_y]):
                mae[pred_x][pred_y] = 1
            elif (y_pred[pred_x][pred_y]  <  y_label[pred_x][pred_y]):
                mae[pred_x][pred_y] = -1
            else:                                         ## else does not matter as such...because pred and y can never be equal
                mae[pred_x][pred_y] = 0                 ## but else -1 destroys everything

    return mae




def feed_forward(X, params):

    cache = {}

    cache["Z1"] = np.matmul(params["W1"], X)   + params["b1"]
    cache["A1"] = sqish(cache["Z1"])
  #  cache["A1"] = sigmoid(cache["Z1"])
  #  cache["A1"] = relu(cache["Z1"])

   # cache["A1"] = x3(cache["Z1"])
#    cache["A1"] = expon_activ(cache["Z1"])

    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]
#    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)          ### softmax

#    cache["A2"] = relu(cache["Z2"])
    cache["A2"] = sigmoid(cache["Z2"])

    return cache





seed_value = 10

#np.random.seed(138)
np.random.seed(seed_value)

# hyperparameters
n_x = X_train.shape[0]
n_h = 5                  ## 200 is ideal for 8x8 digits  ##no. of hidden units >= output classes
#13
#n_h = 16


## batch size = 1 ... bad results... dn know why/

learning_rate = 1/3               # changed by ALI
#learning_rate = 2               # changed by ALI
beta = 0.9
#beta = 0.0000005        ## this value is fantastic for 8x8
#beta = 0                  ## beta is crucial  ## beta = 0 works fine for 8x8 digits   ##  but beta = 1 is impossible

batch_size = 100         # ALI ## depends on data size ## smaller batch --> same accuracy, quick convergence...I think SLOW

#batch_size = 1
batches = -(-m // batch_size)

## division by ZERO ... if batch = 1

#weight_factor = 64      ###
weight_factor = 1           ## 1 -- traditional...

# initialization
params = { "W1": np.random.randn(n_h, n_x) * np.sqrt(1. / (n_x * weight_factor)),
           "b1": np.zeros((n_h, 1)) * np.sqrt(1. / (n_x * weight_factor)),
           "W2": np.random.randn(digits, n_h) * np.sqrt(1. / (n_h * weight_factor)),
           "b2": np.zeros((digits, 1)) * np.sqrt(1. / (n_h * weight_factor))
           }



### W1": np.random.uniform(-1.0, 1.0, size=(n_h, n_x)) * np.sqrt(1. / n_x)       ## no improvement

#total_number_of_epochs = 1700
total_number_of_epochs = 4300


#total_number_of_epochs = 570

#total_number_of_epochs = 85




#np.save('cost_sigmoid_10_epochs', cost_reg)

np.save('weights_1_norm', params["W1"])
np.save('weights_2_norm', params["W2"])
np.save('biases_1_norm', params["b1"])
np.save('biases_2_norm', params["b2"])





print('max. weight: ', np.max(params["W1"]))
print('min. weight: ', np.min(params["W1"]))


#weight_file = open('weights.txt', np.array2string(params["W1"]))

print("Done.")

plot_cost(cost_reg, total_number_of_epochs)         ### plot the test cost
plot_accuracy(accuracy_batches_reg, total_number_of_epochs)




#params["W1"] = np.load('weights.npy')          ## works fine



cache = feed_forward(X_test, params)

predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_test, axis=0)

accuracy_exact = exact_accuracy(predictions, labels)

print(classification_report(predictions, labels))

print('EXACT ACCURACY: ', accuracy_exact)
