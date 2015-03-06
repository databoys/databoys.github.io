---
layout: post
title: Using and improving our neural network (96% MNIST)
published: true
---

Now that we have created our artisan handcrafted neural network we should improve it with some modern techniques that a bunch of really smart people discovered. When I was making these improvements I used the kaggle competition on the MNIST dataset for my benchmarks. That way I could compare my performance to that of other peoples neural networks. I could also check my scores against other tried and true methods listed [here](http://yann.lecun.com/exdb/mnist/). The original neural network that I created for the [last post](http://databoys.github.io/Feedforward/) got 86% on the full MNIST dataset and this new one gets 96%, which is right in line with the multilayer perceptron benchmarks on LeCun's website and [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf). And I am very happy about that! There is still a lot of room for improvement but starting with 96% without using any feature engineering or deep learning techniques is very encouraging. 

![Nice!](http://i.imgur.com/ciWzoO3.png "Nice!")

The new code can be found [here](https://github.com/FlorianMuellerklein/Machine-Learning/blob/master/MultiLayerPerceptron.py). I will go over the improvements as they show up if you were to scroll through the script from top to bottom so it should be easy to follow along. One thing that I won't talk about here is the optimization via numpy. What that means is that I took out a lot of the for loops and replaced them with numpy functions like, numpy.dot() or just used + - * on the arrays directly because numpy will take care of the looping internally. It helps speed things up slightly, most importantly it will make it easier to port the code to [gnumpy](http://www.cs.toronto.edu/~tijmen/gnumpy.html) in order to use a GPU.

One thing to keep in mind is that most of these improvements have the effect of keeping the weights low (closer to 0). For the same reason that regularization works for regression, having low weight values in a neural network can help it generalize better. Since it's very easy to overfit with a neural network anything we'll take whatever we can. 

#####Activation functions#####

The first thing that I did was add two more activation (transfer) functions that we can use. Each one has certain advantages over the logistic sigmoid that we started with. The biggest improvement came from changing the hidden layer activation function from the logistic sigmoid to the hyperbolic tangent. Both are considered sigmoid functions but the logistic is a range of (0, 1) and the hyperbolic tangent (tanh) has a range of (-1, 1). The idea here is that since the tanh function is centered around 0 the outputs that it produces will, on average, be closer to 0. The outputs from the logistic sigmoid will always be greater than 0 so the mean of the outputs will also be greater than 0. 

The next activation function is called softmax. This one is only beneficial in the output layer and only when the classes are mutually exclusive. It forces the output of the neural network to sum to 1, so that they can represent the probability distribution across the classes. This way the network 'knows' that it can't give equal probability to the classes in it's output. Pretty neat!


``` python
def softmax(w):
    e = np.exp(w - np.amax(w))
    dist = e / np.sum(e)
    return dist

def tanh(x):
    return np.tanh(x)
    
def dtanh(y):
    return 1 - y*y
```

The best part is that we can swap these activations into our back propagation algorithm with very few changes. In order to use the tanh function in our hidden layer all we have to do is swap it out for the sigmoid. 

``` python
sum = np.dot(self.wi.T, self.ai)
self.ah = tanh(sum)
```

When we calculate the gradient for the tanh hidden units we will just use the new tanh derivative that we defined earlier in place of the logistic sigmoid derivative. 

``` python
error = np.dot(self.wo, output_deltas)
hidden_deltas = dtanh(self.ah) * error
```

To use the softmax output layer we will make the most drastic changes. We will go from this

``` python
output_deltas = dsigmoid(self.ao) * -(targets - self.ao)
```

to this

``` python
output_deltas = -(targets - self.ao)
```

Again, I am skipping the math in these posts and just focusing on readable python code and a higher level of understanding. But, essentially this is what is happening: if you were to work out the gradient descent algorithm with the derivative of the softmax function you will end up cancelling terms and arrive at -(t - yhat) for the error calculation, where t is the true value and yhat is the predicted value. Awesome!

If I remember right just switching out these activation functions gave me a few percentage points of improvement.

#####Initializing weights#####

In our previous neural network we simply initialized the weights with some random numbers. Which is good because it breaks the symmetry but there is a still a better way. We want to try and activate the sigmoid functions in their linear region so that their derivatives provide enough gradient for our learning to continue. In other words, if the output of a unit is close to the minimum or maximum of the sigmoid function it's derivative will be flat and our network will learn really slowly (there will be no gradient to descend). So how do we fix this?

This part requires some 'coordination' between the input data and the weights for it to be effective. For the input data, we just need to scale them to have a mean of 0. So then we will draw the weights randomly again but this time we will tell numpy to give them a mean of 0 and a standard devation of the negative square root of the size of the layer feeding into the node.

``` python
input_range = 1.0 / self.input ** (1/2)
output_range = 1.0 / self.hidden ** (1/2)
self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden))
self.wo = np.random.normal(loc = 0, scale = output_range, size = (self.hidden, self.output))
```

#####Shuffling training examples#####

This next tip was probably the most simple and effective improvement in the code. On each iteration during training we will now shuffle the order of the data before it is fed into the network. Networks learn the fastest from the most unexpected sample. Lets say that all of our data was neat and organized. All of our 'ones', 'twos', and 'threes' were grouped together. If we fed the data into the network like this it will get really good at classifying 'ones', but then once it gets it's first 'two' it will have no way of even getting close to classifying it. The network will then have to start learning 'twos' and forget about 'ones'. If we randomize the inputs on every iteration the network will have an easier time creating weights that can generalize between all of the classes. 

Adding this to our code is as easy as ... 

``` python
import random 

def fit(self, patterns):
                
    for i in range(self.iterations):
        error = 0.0
        random.shuffle(patterns)
        for p in patterns:
			feed_forward(X)
			backprop_function(y)
```

#####No more overfitting!#####

So there are the three things that have greatly improved the performance of my neural network. Obviously there is still a lot that can be added but these offer pretty big improvements for very little effort. 

Just like the last neural network post, I did not go into the math behind all of this. If you would like to take your understanding of neural networks to the next level the [Stanford deep learning tutorial](http://ufldl.stanford.edu/tutorial/) is my favorite website right now. It offers a much more indepth look at all of the algorithms for neural networks than my posts here. I find it very helpful to match each equation in the 'Multi-Layer Neural Network' tutorial with each snippet of code in my neural network script.

Additionally, many more ways to improve the training of neural networks are outlined in ['Efficient Backprop'](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) by LeCun et al. Most of what I outlined here came from that paper. 

The biggest takeway from all of these tips is that a mean near zero will make you a hero.

#####My machine learning library#####

I am in the process of creating a [machine learning library](https://github.com/FlorianMuellerklein/Machine-Learning) geared toward new users. It will not be the fastest library but the idea is to write the code in a very clear and easy to understand way so that people can go through and see exactly what each algorithm is doing. Hopefully it will be a resource that some people find useful. 