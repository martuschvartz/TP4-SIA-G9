import numpy as np


# we will implement a linear simple perceptron
# that updates using Oja's rule

class SimplePerceptron:
    def __init__(self,input_len, learning_rate=0.001):
        self.input_len = input_len
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_len)  # initialize weights randomly between (0,1)


    def predict(self,X) -> float:
        # compute the output of the perceptron
        return np.dot(self.weights, X)


    def update_weights(self, X):
        # update weights using Oja's rule
        output = self.predict(X)
        for i in range(self.input_len):
            self.weights[i] += self.learning_rate * output * (X[i] - output * self.weights[i])

        #self.learning_rate *= 0.999  # decay learning rate over time
