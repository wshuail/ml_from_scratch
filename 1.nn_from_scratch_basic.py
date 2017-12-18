#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import numpy as np

num_examples = 10  # abbreviate by 'ne' in comments, same below
n_inputs = 5  # ni
n_outputs = 1  # no
lr = 0.03

x = np.linspace(-10, 10, num_examples*n_inputs).astype(np.float32).reshape(-1, n_inputs)
y = np.random.rand(num_examples, n_outputs)


w = np.random.rand(n_inputs, n_outputs)
b = np.random.rand(n_outputs)


def sigmoid_prime(z):
    # Derivative of sigmoid function
    return np.exp(-z)/((1+np.exp(-z))**2)


for i in range(100):
    z = np.dot(x, w) + b  # (nx, no)
    a = 1/(1 + np.exp(-z))  # (nx, no)

    loss = np.sum((a - y)**2)/2

    delta3 = -(y-a)*sigmoid_prime(z)  # (nx, no)*(nx, no) ==> (nx, no)
    dw = np.dot(x.T, delta3)  # (nx, ni).T DOT (nx, no) ==> (ni, no) same with w
    db = np.sum(delta3, axis=0)  # (no,)

    w -= lr*dw
    b -= lr*db

    print('epoch %d loss %s.' %(i, loss))







