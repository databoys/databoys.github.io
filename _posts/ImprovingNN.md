---
layout: post
title: Using and improving our neural network (96% MNIST)
published: false
---

Now that we have created our artisan handcrafted ['neural network'](http://databoys.github.io/Feedforward/) we should improve it with some modern techniques that a bunch of really smart people discovered. When I was making these improvements I used the kaggle competition on the MNIST dataset as my benchmark. That way I could compare my neural net performance to that of other peoples neural networks. I could also check my scores against other tried and true methods listed ['here'](http://yann.lecun.com/exdb/mnist/). The original neural network that I created for the last post got 86% on the full MNIST dataset and the new one with the improvements that I will go over here got 96%. Which is right in line with the benchmarks on LeCun's website that I linked earlier. Which I am very happy about! There is still a lot of room for improvement but starting with 96% without using any feature engineering or deep learning techniques is very encouraging. 

![Nice!](http://i.imgur.com/ciWzoO3.png "Nice!")

The new code can be found ['here'](https://github.com/FlorianMuellerklein/Machine-Learning/blob/master/MultiLayerPerceptron.py). I will go over the improvements as they show up if you were to scroll through the script from top to bottom so it should be easy to follow along. One thing that I won't talk about here is the optimization via numpy. What that means is that I took out a lot of the for loops and replaced them with numpy functions like, numpy.dot() or just used + - * on the arrays directly because numpy will take care of the looping internally. It helps speed things up slightly, most importantly it will make it easier to port the code to ['gnumpy'](http://www.cs.toronto.edu/~tijmen/gnumpy.html) in order to use a GPU.

One thing to keep in mind is that most of these improvements have the effect of keeping the weights low. For the same reason that regularization works for regression, having low weight values in a neural network can help it generalize better. It's very easy to overfit with a neural network so anything that we can do to help with that will be very nice.

#####Activation functions#####

The first thing that I did was add two more activation (transfer) functions that we can use. Each one has certain advantages over the logistic sigmoid that we started with. The biggest improvement came from changing the hidden layer activation function from the logistic sigmoid to the hyperbolic tangent. Both are considered sigmoid functions but the logistic is a range of (0, 1) and the hyperbolic tangent (tanh) has a range of (-1, 1). 