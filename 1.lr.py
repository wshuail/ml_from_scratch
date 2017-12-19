#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import numpy as np

num_examples = 10
n_inputs = 1
n_outputs = 1
lr = 0.03

x = np.linspace(-10, 10, num_examples).astype(np.float32).reshape(num_examples, -1)
y = 3*x + 2 + np.random.rand(*x.shape)

w = np.random.rand(num_examples, 1)
b = np.random.rand(num_examples, 1)


for i in range(100):
    z = x*w + b

    loss = z - y
    cost = np.sum(loss**2)/2

    delta3 = -(y-z)
    dw = x*loss/num_examples
    db = np.sum(loss)/num_examples

    w -= lr*dw
    b -= lr*db

    if i % 10 == 0:
        print('epoch %d cost %s.' % (i, cost))







