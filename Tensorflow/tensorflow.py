# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:14:46 2020

@author: Admin
"""

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))