"""
Author: Israel
Student ID: 1675980
Date: March 12 , 2024

This script implements logistic regression, incorporating forward pass, backpropagation, and a gradient descent
algorithm. It initializes inputs with random weights and a bias, performs forward pass to calculate the loss,
and updates the weights through backpropagation. Following several iterations, the model's accuracy is evaluated."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def sigmoid_function(X):
    return 1 / (1 + np.exp(-X))

def initialize(X):

    row, col = X.shape
    weights = np.random.randn(col)
    bias = 0.3
    return weights, bias

def forward_pass(X, weights, bias):
    z = np.dot(X, weights) + bias
    y_predict = sigmoid_function(z)
    return y_predict

def back_propagation(X,y_predict, y_actual ):
    # this is the derivative of the binary loss entropy function which utilizes the chain rule
    # with the following dl/dy_predict * dy_predict/d sigmond output * the input
    derivative_weight1 = -((y_actual/y_predict) - ((1 - y_actual)/(1-y_predict))) * ((y_predict)*(1-y_predict)) * X[0]
    derivative_weight2 = -((y_actual/y_predict) - ((1 - y_actual)/(1-y_predict))) * ((y_predict)*(1-y_predict)) * X[1]
    derivative_bias = -((y_actual/y_predict) - ((1 - y_actual)/(1-y_predict))) * ((y_predict)*(1-y_predict))
    return derivative_weight1, derivative_weight2, derivative_bias

def training(learning_rate, n_iterations, X, y):
    row, col = X.shape
    total_loss = 0
    weights, bias = initialize(X)
    for n in range(n_iterations):
        for i in range(len(X)):
            #SGD
            y_predict = forward_pass(X[i], weights, bias) # error from forward pass
            total_loss += binary_cross_entropy_loss(y[i], y_predict)
            # print(f"Iteration :" + str(i)+ ", Loss: "+ str(loss))
            d_weight1, d_weight2 ,d_bias = back_propagation(X[i], y_predict, y[i])
            print(d_weight1)
            #updating weights and bias using
            weights[0] = weights[0] - (learning_rate * d_weight1)
            weights[1] = weights[1] - (learning_rate * d_weight2)
            bias = bias - (learning_rate * d_bias)
        print(f"Iteration :" + str(n)+ ", Total Loss: "+ str(total_loss / len(X)))
        #average total loss in each iteration
        total_loss = 0

    return weights, bias

def binary_cross_entropy_loss(y_actual, y_predict):
    y1 = y_actual * np.log(y_predict)
    y2 = (1 - y_actual) * np.log(1 - y_predict)
    return -(y1 + y2)

def predictions(X, weights, bias):
    predictions = np.array([])
    z = np.dot(X, weights) + bias
    y_pred = sigmoid_function(z)
    for pred in y_pred:
        if pred >= 0.5:
            predictions = np.append(predictions, 1)
        else:
            predictions = np.append(predictions, 0)
    return predictions

def evaluate_accuracy(predictions, y_test):
    accuracy = np.sum(predictions == y_test) / len(y_test)
    return accuracy

if __name__ == '__main__':
    # Generate synthetic data
    num_samples_per_class = 100
    np.random.seed(0)
    X1 = np.random.randn(num_samples_per_class, 2) + np.array([2, 2])
    X2 = np.random.randn(num_samples_per_class, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.array([0] * num_samples_per_class + [1] * num_samples_per_class)

    # Split data into training and testing sets
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    weights, bias = training(0.1, 100, X_train, y_train)

    # Make predictions
    y_predictions = predictions(X_test, weights, bias)

    # Evaluate accuracy

    print(f"Accuracy: ", accuracy_score(y_predictions, y_test))









