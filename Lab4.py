"""
Author: Israel
Student ID: 1675980
Date: March 3 , 2024

This code includes a perceptron class within this class we will have a method for fitting the data, an activation function,
for the summation of the weights, a method to update the weights and bias, then lastly we have the predict function used
to compute the predicted output.
 """

import numpy as np
from sklearn.model_selection import train_test_split

class Preceptron:

    def __init__(self,n , a ):

        self.n_iterations = n # math symbol for number of iterations
        self.learning_rate = a #math symbol for learning rate
        self.Weight = None
        self.Bias = None


    def activation_function(self,summation ):
        """This method is used get the predicted output based on the value from the summation of weights* input subtracted
        by the bais"""
        if(summation > 0):
            return 1
        return 0

    def update(self,input, actual_output, predicted_output, learning_rate):
        "this method is used to update the weights and bias with a formula called the perceptron learning rule"
        preceptron_learning = learning_rate*(actual_output - predicted_output)
        self.Weight +=   preceptron_learning * input
        self.Bias += preceptron_learning


    def fit(self, X_input, Y_output):
        """"this method intializes the weights and bias to random values, however I thought about using zero. however
        it didn't change the result when i adjusted the learning rate and iterations. also this function is used
        to train the data by using predict which contains a calculation that get the outputs"""
        col = X_input.shape[1]
        self.Bias = 0.5
        # self.Weight = np.zeros(shape=col)
        self.Weight = np.array([0.0,0.0])
        for iteration in range(self.n_iterations):

            for input_index, input in enumerate(X_input):
                predicted_output = int(self.predict(input))
                self.update(input, Y_output[input_index], predicted_output, self.learning_rate)
        print(f"Bais:  {self.Bias}")
        print(f"Weight {self.Weight}")




    def predict(self, X_input):
        "this method predict by getting the summations and then using an activation function to return the binary output"
        summation = np.dot(X_input, self.Weight) + self.Bias
        predicted_output = self.activation_function(summation)
        return predicted_output

"Intialize the inputs and outputs to train the perceptron"
X_inputs = np.array([[0, 0], [1, 0], [1, 0], [1, 1]])
Y_Output = np.array([0,1,1,0])

perceptron = Preceptron(100, 0.3)
perceptron.fit(X_inputs,Y_Output)

"This will be when i test the model using unseen data"
test_inputs = np.array([[0, 0], [1, 0], [1, 0], [1, 1]])
for inputs in test_inputs:
    prediction = perceptron.predict(inputs)
    print(f"[{inputs[0]}, {inputs[1]}] --> {prediction}")

inputs = np.array([1,0])
bais = -0.1
weights = np.array([0.3,-0.6])

answer = np.dot(inputs,weights) + bais
print(f"answer: {answer}")