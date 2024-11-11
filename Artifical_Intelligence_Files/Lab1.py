'''
Author: Israel
Student ID: 1675980
Date: February

This program is a custom implementation of the K-Nearest Neighbors (KNN) algorithm, designed to predict the likelihood
of diabetes in patients based on various health metrics. It utilizes a dataset that includes features such as
'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', and 'Age'.
The 'y' column in the dataset denotes the label, where 1 indicates the presence of diabetes and 0 indicates its absence.

The KNN algorithm in this program is set to consider the three nearest neighbors (k=3) for its predictions.
The program reads the data from a CSV file, processes the features, and splits the dataset into training and
testing sets. It then standardizes the features to ensure that they contribute equally to the prediction process.

The KNN classifier in this program is implemented from scratch, relying only on the numpy library for
mathematical operations. The Euclidean distance is used as the distance metric to determine the 'nearness'
of neighbors. The classifier's performance is evaluated on the testing set using the classification report from
sklearn, providing insights into its precision, recall, f1-score, and support.

I used this to help me build this classifier
"https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/"
'''

import math
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to calculate standard deviation
def standardDeviation(array, outerIndex,n):
    """Calculate the standard deviation of a given array."""
    std = 0
    mean = sum(array[outerIndex]) / n
    for innerIndex in range(len(array[0])):
        std = std + (array[outerIndex][innerIndex] - mean) ** 2
    return math.sqrt(std/n)

# Function to standardize an array
def stand(array, std, outerIndex,n):
    """Standardize a given array using the provided standard deviation and mean."""
    mean = sum(array[outerIndex]) / n
    for innerIndex in range(len(array[0])):
        array[outerIndex][innerIndex] = (array[outerIndex][innerIndex] - mean) / std

# Function to complete the sum operation
def finishedSum(array):
    """Complete the sum operation for a given array."""
    for outerIndex in range(len(array)):
        deviation = standardDeviation(array, outerIndex)
        stand(array, deviation, outerIndex)
    return array

# Function to calculate Euclidean distance
def euclidean_distance(row1, row2):
    """Calculate the Euclidean distance between two data points."""
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

# Function to append distances in an array
def AppendDistanceInArray(X_train, data):
    """Adds the calculated distance from a single iteration over a row of
     the test dataset (X_test) to the list of distances. """
    distances = []
    for feature in X_train:
        distances.append(euclidean_distance(feature,data))
    return distances

# Function to find k indices
def sortDistances(distances, k):
    """Find the indices of the k nearest neighbors(3) by sorting the distance array then returning a list of 3
    indices"""
    sorted_distances_indexes = np.argsort(distances)[:k]
    return sorted_distances_indexes

# Function to map indices to labels
def mappedToLabels(indices, Y_train):
    """Map the indices of the nearest neighbors to their corresponding labels."""
    k_labels =[]
    for index in indices:
        k_labels.append(Y_train[index])
    return k_labels

# Function to find majority vote
def majorityVote(mappedLabels):
    """Find the majority label among the nearest neighbors list of 3 labels"""
    counter0 = 0 # label 0 occurences
    counter1= 0 # label 1 occurences
    for labels in mappedLabels :
        if labels == 0:
            counter0+=1
        else:
            counter1+=1
    #find labels has more occurences and this will be used as the majority label
    if counter0 > counter1:
        return 0
    else:
        return 1

# Function to implement the K-Nearest Neighbors algorithm
def KNearestNeighbor(X_train,X_test, Y_train, k):
    """Implement the K-Nearest Neighbors algorithm. returns the predictions for the test set.
    data is the vectors of the X_test"""
    predictions = []
    for data in X_test:
        distances = AppendDistanceInArray(X_train, data)
        sortedIndices = sortDistances(distances, k)
        mappedLabels = mappedToLabels(sortedIndices,Y_train)
        predictions.append(majorityVote(mappedLabels))
    return predictions

# Function to get all labels and features from the dataset
def getAllLabelsAndFeatures():
    """Get all labels and features from the dataset. this function reads the data and converts it to a data frame
    y will be the labels and x will be the features, then we will split each of them to test and training data. 20% test
    and 80% train"""
    df = pd.read_csv('diabetes.csv')
    y = df['y'].values
    X = df.drop(['y'], axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to standardize the training and testing sets
def standardize(X_train, X_test):
    """Standardize the training and testing sets."""
    X_train = (X_train - X_train.mean())/X_train.std()
    X_test = (X_test-X_test.mean())/X_test.std()
    return X_train,X_test

# Main function
if __name__ == "__main__":
    """Main function to run the K-Nearest Neighbors algorithm and print the classification report."""
    X_train, X_test, y_train, y_test = getAllLabelsAndFeatures()
    X_train,X_test = standardize(X_train, X_test)
    #k: the number of data we will choose from when doing the majority vote,this will determine which label is closest
    #to some unknown label
    Y_pred = KNearestNeighbor(X_train, X_test, y_train, 3)
    print(classification_report(y_test, Y_pred))

