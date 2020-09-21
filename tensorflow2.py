import tensorflow as tf
import numpy as np
import pdb
with tf.name_scope('placeholders'):
    x = tf.placeholder('float', [None, 1])
    y = tf.placeholder('float', [None, 1])

with tf.name_scope('neural_network'):
    x1 = tf.contrib.layers.fully_connected(x, 100,activation_fn=tf.keras.activations.tanh)
    x2 = tf.contrib.layers.fully_connected(x1, 100,activation_fn=tf.keras.activations.tanh)
    result = tf.contrib.layers.fully_connected(x2, 1,
                                               activation_fn=None)

    loss = tf.nn.l2_loss(result - y)
with tf.name_scope("variable"):


     W1 = tf.get_variable("W1", [100,1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
     b1 = tf.get_variable("b1", [100,1], initializer = tf.zeros_initializer())
     W2 = tf.get_variable("W2", [100,100], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
     b2 = tf.get_variable("b2", [100,1], initializer = tf.zeros_initializer())
     W3 = tf.get_variable("W3", [1,100], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
     b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())



with tf.name_scope('optimizer'):
    train_op = tf.train.AdamOptimizer().minimize(loss)

import math
xpts = np.random.rand(256)
xpts -= 0.5
xpts *= math.pi
ypts = np.sin(xpts)
xpts2 = xpts.reshape(256,1)
print(True)
parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
ypred = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Train the network
    for i in range(5000):


        _, loss_result = sess.run([train_op, loss],
                                  feed_dict={x: xpts[:, None],
                                             y: ypts[:, None]})
        if(i % 1000 == 0):
            print('iteration {}, loss={}'.format(i, loss_result))


#     xpts2 -= 0.5
#     xpts2 *= math.pi
    paramtrs = sess.run(parameters, feed_dict={x: xpts2})
    W1 = paramtrs["W1"]
    W2 = paramtrs["W2"]
    W3 = paramtrs["W3"]
    b1 = paramtrs["b1"]
    b2 = paramtrs["b2"]
    b3 = paramtrs["b3"]
    z1 = np.dot(W1,xpts2.T) + b1
    a1 = np.tanh(z1)
    print(a1.shape,W2.shape,W3.shape,b2.shape,b3.shape)
    z2 = np.dot(W2,a1) + b2
    a2 = np.tanh(z2)
    ypred= a2
    a3 = np.dot(W3,a2) + b3

    ypred = a3
    print(True)

print(ypred[0][0])
print(np.sin(xpts2[0]))
