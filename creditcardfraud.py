#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:50:00 2019

@author: heroo
"""

import tensorflow as tf
import numpy as np
#hello = tf.constant("Hello, world!")
sess = tf.Session()
#print(sess.run(hello))


n_f =10
n_d_n = 3
x=tf.placeholder(tf.float32,(None,n_f))

b=tf.Variable(tf.zeros([n_d_n]))
w=tf.Variable(tf.random_normal([n_f,n_d_n]))

xw=tf.matmul(x,w)
z=tf.add(xw,b)
a=tf.sigmoid(z)
init=tf.global_variables_initializer()
sess.run(init)
print(sess.run(a,feed_dict={x:np.random.random([1,n_f])}))



