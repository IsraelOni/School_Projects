"""
Author: Israel
Student ID: 1675980
Date: March 12 , 2024

This code contains functions that aim to find a solution for the amount of colors used for a picture.
this algorithm is meant to find the closest colors to all the colors representing this image.
by doing this we simpify the representation for the image.
 """


import numpy as np
import matplotlib.pyplot as plt
import cv2
import math



"""The function accepts the input image and pre-process and 
it Sets the range of the pixels between 0 and 1
. Also transforms the the matrix to a two-dimensional matrix"""
def preProcessData(image):

    image = image / 255.0
    height, weight , colors = image.shape
    matrix = image.reshape(-1, colors)
    return matrix

def euclidean_distance(row1, row2):
    """Calculate the Euclidean distance between two data points."""
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2

    return math.sqrt(distance)

    # Function to append distances in an array
"""This function takes the input image and number ofcentroids K 
and returns the list of centroids."""
def kMeans_init_centroid(input , K):
    pixel, colors = input.shape
    centroids = np.zeros(shape =(K, colors))
    for i in range(K):
        centroids[i] = input[np.random.randint(0, pixel+1), :]

    return centroids



"""return the new centroid by computing the
# means of the data points assigned to each centroid. Agrs: X: (m, n)
# data points, idx: (m,) array containing the index of the closest
# centroid for each example in X. Concretely, idx[i] contains the index
# of the centroid closest to example i, K represents the number of
# centroids. The function returns the new centroid (k,n)."""
def compute_centoids(idx , input, K):
    pixel, colors = input.shape
    centroids = np.zeros(shape = (K, colors))
    count = np.zeros(shape = (K,1))
    for i in range(pixel):
        centroids[idx[i].astype(int)] += input[i]
        count[idx[i].astype(int)] += 1
    return centroids/count




"""Computes the centroid memberships for every example. It takes the input matrix X and centroids. It should
output a one-dimension array idx which has the same number of elements
 as X that holds the index of the closest centroid. Args: X
(m,n) input values, centroids: k centroids. Returns idx(m,) closest centroids."""
def find_closest_centroid(input_X , centroids):
    distance = 0.0
    idx = np.zeros(len(input_X))
    for i in range(len(input_X)):
        min_distance = 100
        for j in range(len(centroids)):
            distance = np.sqrt(np.sum( (input_X[i, :] - centroids[j, :])**2))
            if distance <= min_distance:
                min_distance = distance
                idx[i] = int(j)



    return idx
"""takes the pre-processed image, centroids and max_iter and
# returns the centroids, idx (i.e., the corresponding index). You need the
# following helper functions which are find_closest_centroid and compute_centoids"""
def runKmeansAlgorithm(centroids, imageInput, max_iter):
    k = 16
    for i in range(max_iter):
        idx = find_closest_centroid(imageInput, centroids)
        centroids = compute_centoids(idx, imageInput, k)
    return centroids, idx

"""After finding the top K=16 colors to represent the image, 
you can now assign each pixel position to its closest centroid."""
def compress_image(centroid, idx, img):

    centroid = centroid[idx.astype(int), :]
    centroid3d = centroid.reshape(img.shape)
    print(centroid3d)


    plt.imshow(centroid3d)
    plt.show()


"""this is where i run most of my code"""
if __name__ == '__main__':
    img = cv2.imread('assignment_1.png')  # Replace with the actual path to your image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = preProcessData(img)
    centriods = kMeans_init_centroid(image, 16)
    centroids,idx = runKmeansAlgorithm(centriods, image, 10)
    compress_image(centriods, idx, img)







