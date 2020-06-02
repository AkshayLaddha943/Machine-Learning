# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:47:18 2020

@author: Admin
"""

import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

W = tf.Variable([-0.5], dtype = tf.float32)
b = tf.Variable([0.5], dtype = tf.float32)




x = tf.placeholder(dtype = tf.float32)
y = tf.placeholder(dtype = tf.float32)

x_train = [1,2,3,4]
y_train = [0 , -1, -2, -3]


linear_model = W * x + b
loss = tf.reduce_sum(tf.square(linear_model - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

session =  tf.compat.v1.Session()
init = tf.global_variables_initializer()
session.run(init)

for i in range(500):
    session.run(train, {x:x_train, y : y_train})

print(session.run([W, b, loss], {x:x_train, y : y_train }))