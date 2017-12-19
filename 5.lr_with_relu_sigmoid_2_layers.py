#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import numpy as np

num_examples = 10  # abbreviate by 'ne' in comments, same below
n_inputs = 5  # ni
n_hidden = 3  # nh
n_outputs = 1  # no
lr = 0.03

x = np.linspace(-10, 10, num_examples*n_inputs).astype(np.float32).reshape(-1, n_inputs)
y = np.random.rand(num_examples, n_outputs)


w1 = np.random.rand(n_inputs, n_hidden)
b1 = np.random.rand(1, n_hidden)

w2 = np.random.rand(n_hidden, n_outputs)
b2 = np.random.rand(1, n_outputs)


def relu_derivative(x):
    return 1.0*(x > 0)


def sigmoid_prime(z):
    # Derivative of sigmoid function
    return np.exp(-z)/((1+np.exp(-z))**2)


for epoch in range(100):
    z1 = np.dot(x, w1) + b1
    # a1 = 1/(1 + np.exp(-z1))  # a1: (nx, nh)
    a1 = z1*(z1 > 0)

    z2 = np.dot(a1, w2) + b2
    a2 = 1/(1 + np.exp(-z2))

    loss = np.sum((a2 - y)**2)/2

    delta2 = -(y-a2)*sigmoid_prime(a2)  # (nx, no)*(nx, no) ==> (nx, no)
    dw2 = np.dot(a1.T, delta2)/num_examples  # (nh, nx) DOT (nx, no) ==> (nh, no)  same with w2
    db2 = np.sum(delta2, axis=0, keepdims=True)/num_examples  # (nx, no) ==> (1, no) same with b2

    delta1 = np.dot(delta2, w2.T)*relu_derivative(a1)  # ((nx, no) DOT (nh, no).T) * (nx, nh) ==> (nx, nh)
    dw1 = np.dot(x.T, delta1)/num_examples  # (ni, nx) DOT (nx, nh) ==> (ni, nh)  same with w1
    db1 = np.sum(delta1, axis=0, keepdims=True)/num_examples  # (nx, nh) ==> (1, nh) same with b2

    w2 -= lr*dw2
    b2 -= lr*db2
    w1 -= lr*dw1
    b1 -= lr*db1

    if epoch % 10 == 0:
        print('epoch %d loss %s.' % (epoch, loss))







