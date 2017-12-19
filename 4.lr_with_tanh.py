#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import numpy as np

num_examples = 10  # abbreviate by 'ne' in comments, same below
n_inputs = 5  # ni
n_outputs = 1  # no
lr = 0.3

x = np.linspace(-10, 10, num_examples*n_inputs).astype(np.float32).reshape(-1, n_inputs)
y = np.random.normal(loc=0, scale=1, size=(num_examples, n_outputs))


w = np.random.rand(n_inputs, n_outputs)
b = np.random.rand(1, n_outputs)


def tanh_prime(z):
    return 1 - np.square(z)


for i in range(100):
    z = np.dot(x, w) + b  # (nx, no)
    a = np.tanh(z)

    loss = np.sum((a - y)**2)/2

    delta3 = -(y-a)*tanh_prime(a)  # (nx, no)*(nx, no) ==> (nx, no)
    dw = np.dot(x.T, delta3)/num_examples  # (nx, ni).T DOT (nx, no) ==> (ni, no) same with w
    db = np.sum(delta3, axis=0, keepdims=True)/num_examples  # (no,)

    w -= lr*dw
    b -= lr*db

    if i % 10 == 0:
        print('epoch %d loss %s.' % (i, loss))




