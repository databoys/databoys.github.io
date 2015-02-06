---
layout: post
title: Neural Network with numpy
published: true
---

Import modules and set up equations.

```python
import numpy as np
np.seterr(all = 'ignore')

# sigmoid transfer function
# IMPORTANT: when using the logit (sigmoid) transfer function make sure y values are scaled from 0 to 1
# if you use the tanh then you should scale between -1 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def dsigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))
```

We will make the neural network into a class so it can be used like a sklearn module and initialize the layers. 

```python
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
        
```

The very first thing that a neural network does is push the input data through the network using the randomized weights that we set up when we initialized. We will call that 'feedForward' because we are feeding the data forward through the network with the current weights. 

```python
    def feedForward(self, inputs):
        """
        The feedforward algorithm loops over all the nodes in the hidden layer and
        adds together all the outputs from the input layer * their weights
        the output of each node is the sigmoid function of the sum of all inputs
        which is then passed on to the next layer.
        :param inputs: input data
        :return: updated activation output vector
        """
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
 ```
 