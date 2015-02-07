---
layout: post
title: Neural Network with numpy
published: true
---

Neural networks are a pretty badass machine learning algorithm for classification. For me, they seemed pretty intimidating to try to learn but when I finally buckled down and got into them it wasn't so bad. They are called neural networks because they are loosly based on how the brain's neurons work but they are essentially a group of linear models. There is a lot of good information about the math and structure of these algorithms so I will skip that here. Instead I will outline the steps to writing one in python with numpy and hopefully explain it very clearly. 

First, we can think of every neuron as having an activation function. We will use the sigmoid function, which should be very familiar because of logistic regression. Unlike logistic regression, we will also need the derivative of the sigmoid function when using a neural net. 

'''python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def dsigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))
	
'''

Much like logistic regression, the sigmoid function in a neural network will generate the end point (activation) of inputs multiplied by their weights. For example, lets say we had two columns (features) of input data and one hidden node (neuron) in our neural network. Each feature would be multiplied by its corresponding weight value and then added together and passed through the sigmoid (just like a logistic regression). To take that simple example and turn it into a neural network we just add more hidden units, and every input feature has a 'path' to each hidden unit where it is multiplied by it's corresponding weight. Each hidden unit takes the sum of it's inputs * weights and passes that through the sigmoid resulting in that unit's activation. Hopefully that made sense. 

Next we will set up the network and initialize some parameters. 

'''python
class MLP_NeuralNetwork(object):
    def __init__(self, input, hidden, output):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden
        self.output = output

        # set up array of 1s for activations
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output

        # create randomized weights
        self.wi = np.random.randn(self.input, self.hidden) # weight vector going from input to hidden
        self.wo = np.random.randn(self.hidden, self.output)  # weight vector going from hidden to output

        # create arrays of 0 for changes
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))
'''

We are going to do all of these calculations with matricies because it's much easier to read and it's faster. Our class will take three inputs; the size of the input layer (# features), the size of the hidden layer (variable parameter to be tuned), and the number of the output layer (# of possible classes). We set up an array of 1s as a placeholder for the unit activations and an array of 0s as a placeholder for the layer changes. One important thing to note is that we initialized all of the weights to random numbers. It's important for the weights to be random otherwise we won't be able to tune the network. If all of the weights are the same then all of the hidden units will be the same and you'll be screwed. 

So now it's time to make some predictions. What we will do is feed all of the data forward through the network with the random weights and generate some (bad) predictions. Later, each time the predictions are made we calculate how wrong the predictions are and in what direction we need to change the weights in order to make the predictions better. We will do this many many times as the weights get updated so we'll make a feed forward function that can be called over and over again.

'''python
	def feedForward(self, inputs):
       if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs you silly goose!')

        # input activations
        for i in range(self.input -1): # -1 is to avoid the bias
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]
'''

The input activations are just the input features. But, for each other layer the activations become the sum of the previous layers activations multiplied by their corresponding weights which are then fed into the sigmoid. 