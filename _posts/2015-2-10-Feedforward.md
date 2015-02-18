---
layout: post
title: Neural network with numpy
published: true
---

Neural networks are a pretty badass machine learning algorithm for classification. For me, they seemed pretty intimidating to try to learn but when I finally buckled down and got into them it wasn't so bad. They are called neural networks because they are loosely based on how the brain's neurons work.  However, they are essentially a group of linear models. There is a lot of good information about the math and structure of these algorithms so I will skip that here. Instead I will outline the steps to writing one in python with numpy and hopefully explain it very clearly. The code here is heavily based on the neural network code provided in ['Programming Collective Intelligence'](http://shop.oreilly.com/product/9780596529321.do), I tweaked it a little to make it usable with any dataset as long as the input data is formatted correctly. 

First, we can think of every neuron as having an activation function. This function determines whether the neuron is ‘on’ or ‘off’ – fires or not. We will use the sigmoid function, which should be very familiar because of logistic regression. Unlike logistic regression, we will also need the derivative of the sigmoid function when using a neural net.

``` python
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# derivative of sigmoid
def dsigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))	
```

Much like logistic regression, the sigmoid function in a neural network will generate the end point (activation) of inputs multiplied by their weights. For example, lets say we had two columns (features) of input data and one hidden node (neuron) in our neural network. Each feature would be multiplied by its corresponding weight value and then added together and passed through the sigmoid (just like a logistic regression). To take that simple example and turn it into a neural network we just add more hidden units. In addition to adding more hidden units, we add a path from every input feature to each of those hidden units where it is multiplied by its corresponding weight. Each hidden unit takes the sum of it's inputs * weights and passes that through the sigmoid resulting in that unit's activation. 

Next we will set up the arrays to hold the data for network and initialize some parameters. 

``` python
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
        self.wi = np.random.randn(self.input, self.hidden) 
        self.wo = np.random.randn(self.hidden, self.output) 
        # create arrays of 0 for changes
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))
```

We are going to do all of these calculations with matricies because they are fast and super easy to read. Our class will take three inputs; the size of the input layer (# features), the size of the hidden layer (variable parameter to be tuned), and the number of the output layer (# of possible classes). We set up an array of 1s as a placeholder for the unit activations and an array of 0s as a placeholder for the layer changes. One important thing to note is that we initialized all of the weights to random numbers. It's important for the weights to be random otherwise we won't be able to tune the network. If all of the weights are the same then all of the hidden units will be the same and you'll be screwed. 

So now it's time to make some predictions. What we will do is feed all of the data forward through the network with the random weights and generate some (bad) predictions. Later, each time the predictions are made we calculate how wrong the predictions are and in what direction we need to change the weights in order to make the predictions better (i.e. error). We will do this many, many … MANY times as the weights get updated so we'll make a feed forward function that can be called over and over again.


``` python
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
```

The input activations are just the input features. But, for each other layer the activations become the sum of the previous layers activations multiplied by their corresponding weights fed into the sigmoid. 

On the first pass our predictions will be pretty bad. So we'll use a very familiar concept, gradient descent. This is the part that I get excited about because I think the math is really clever. Unlike gradient descent for a linear model we need to use a little bit of calculus for a neural network. Which is why we wrote the function for the derivative of the sigmoid function at the beginning. 

Our backpropagation algorithm begins by computing the error of our predicted output against the true output. We then take the derivative of the sigmoid on the output activations (predicted values) in order to get the direction (slope) of the gradient and multiply that value by the error. Which gives us the magnitude of the error and which direction the hidden weights need to be changed in order to correct it. We then move on to the hidden layer and calculate the error of hidden layer weights based on the magnitude and error calculated previously. 

Using that error and the derivative of the sigmoid on the hidden layer activations we calculate how much and in which direction the weights need to change for the input layer.

Now that we have the values for how much we want to change the rates and in what direction we move on to actually doing that. We update the weights connecting each layer. We do this by multiplying the current weights by a learning rate constant and the magnitude and direction for the corresponding layer of weights. Just like in linear models we use a learning rate constant to make small changes at each step so that we have a better chance at finding the true values for the weights that minimize the cost function.
 

``` python
def backPropagate(self, targets, N):
	"""
    :param targets: y values
    :param N: learning rate
    :return: updated weights and current error
    """
    if len(targets) != self.output:
        raise ValueError('Wrong number of targets you silly goose!')
    # calculate error terms for output
    # the delta tell you which direction to change the weights
    output_deltas = [0.0] * self.output
    for k in range(self.output):
        error = targets[k] - self.ao[k]
        output_deltas[k] = dsigmoid(self.ao[k]) * error
    # calculate error terms for hidden
    # delta tells you which direction to change the weights
    hidden_deltas = [0.0] * self.hidden
    for j in range(self.hidden):
        error = 0.0
        for k in range(self.output):
            error += output_deltas[k] * self.wo[j][k]
        hidden_deltas[k] = dsigmoid(self.ah[j]) * error
    # update the weights connecting hidden to output
    for j in range(self.hidden):
        for k in range(self.output):
            change = output_deltas[k] * self.ah[j]
            self.wo[j][k] += N * change + self.co[j][k]
            self.co[j][k] = change
    # update the weights connecting input to hidden
    for i in range(self.input):
        for j in range(self.hidden):
            change = hidden_deltas[j] * self.ai[i]
            self.wi[i][j] += N * change + self.ci[i][j]
            self.ci[i][j] = change
    # calculate error
    error = 0.0
    for k in range(len(targets)):
        error += 0.5 * (targets[k] - self.ao[k]) ** 2
    return error
```

Alright, lets tie it all together and create training and prediction functions. The steps to training the network are pretty straight forward and intuitive. We first call the 'feedForward' function which gives us the outputs with the randomized weights that we initialized. Then we call the backpropagation algorithm to tune and update the weights to make better predictions. Then the feedForward function is called again but this time it uses the updated weights and the predictions are a little better. We keep this cycle going for a predeterimined amount of iterations during which we should see the error drop close to 0. 

``` python
def train(self, patterns, iterations = 3000, N = 0.0002):
    # N: learning rate
    for i in range(iterations):
        error = 0.0
        for p in patterns:
            inputs = p[0]
            targets = p[1]
            self.feedForward(inputs)
            error = self.backPropagate(targets, N)
        if i % 500 == 0:
            print('error %-.5f' % error)
```
Finally, for the predict function. We just simply call the feedForward function which will return the activation of the output layer. Remember, the activation of each layer is a linear combination of the output of the previous layer * the corresponding weights pushed through the sigmoid. 

``` python
def predict(self, X):
    """
    return list of predictions after training algorithm
    """
    predictions = []
    for p in X:
        predictions.append(self.feedForward(p))
    return predictions
```

That's basically it! You can see the full code [here](https://github.com/FlorianMuellerklein/Machine-Learning/blob/master/BackPropagationNN.py).

I ran this code on the digit recognition dataset provided by sklearn and it finished with an accuracy of 97%. I'd say that's pretty successful!

```
             precision    recall  f1-score   support

          0       0.98      0.96      0.97        49
          1       0.92      0.97      0.95        36
          2       1.00      1.00      1.00        43
          3       0.95      0.88      0.91        41
          4       0.98      1.00      0.99        47
          5       0.96      1.00      0.98        46
          6       1.00      1.00      1.00        47
          7       0.98      0.96      0.97        46
          8       0.93      0.80      0.86        49
          9       1.00      0.91      0.95        46

avg / total       0.97      0.95      0.96       450
```
