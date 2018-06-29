
# coding: utf-8

# # TensorFlow Assignment: Convolutional Neural Network (CNN)

# **[Duke Community Standard](http://integrity.duke.edu/standard.html): By typing your name below, you are certifying that you have adhered to the Duke Community Standard in completing this assignment.**
# 
# Name: Matthew Epland

# ### Convolutional Neural Network
# 
# Build a 2-layer CNN for MNIST digit classfication. Feel free to play around with the model architecture and see how the training time/performance changes, but to begin, try the following:
# 
# Image -> Convolution (32 5x5 filters) -> nonlinearity (ReLU) ->  (2x2 max pool) -> Convolution (64 5x5 filters) -> nonlinearity (ReLU) -> (2x2 max pool) -> fully connected (256 hidden units) -> nonlinearity (ReLU) -> fully connected (10 hidden units) -> softmax
# 
# Some tips:
# - The CNN model might take a while to train. Depending on your machine, you might expect this to take up to half an hour. If you see your validation performance start to plateau, you can kill the training.
# 
# - Since CNNs a more complex than the logistic regression and MLP models you've worked with before, so you may find it helpful to use a more advanced optimizer. You're model will train faster if you use [`tf.train.AdamOptimizer`](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) instead of `tf.train.GradientDescentOptimizer`. A learning rate of 1e-4 is a good starting point.

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import trange
import numpy as np


# In[2]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);


# Image -> Convolution (32 5x5 filters) -> nonlinearity (ReLU) ->  (2x2 max pool) -> Convolution (64 5x5 filters) -> nonlinearity (ReLU) -> (2x2 max pool) -> fully connected (256 hidden units) -> nonlinearity (ReLU) -> fully connected (10 hidden units) -> softmax

# In[3]:


tf.reset_default_graph()

# input layers
x = tf.placeholder(tf.float32, [None,28*28])
x_2D = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None,10])

# Convolution (32 5x5 filters) -> nonlinearity (ReLU)
W_1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
b_1 = tf.Variable(tf.zeros([32]))
conv_1 = tf.nn.relu(tf.nn.conv2d(x_2D, W_1, strides=[1,1,1,1], padding="SAME") + b_1)

# (2x2 max pool)
max_pool_1 = tf.nn.max_pool(conv_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# Convolution (64 5x5 filters) -> nonlinearity (ReLU)
W_2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
b_2 = tf.Variable(tf.zeros([64]))
conv_2 = tf.nn.relu(tf.nn.conv2d(max_pool_1, W_2, strides=[1,1,1,1], padding="SAME") + b_2)

# (2x2 max pool)
max_pool_2 = tf.nn.max_pool(conv_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# flatten for fully connected layer
flat = tf.reshape(max_pool_2, [-1, 7*7*64])

# fully connected (256 hidden units) -> nonlinearity (ReLU)
W_3 = tf.Variable(tf.truncated_normal([7*7*64, 256], stddev=0.1))
b_3 = tf.Variable(tf.zeros([256]))
fc_1 = tf.nn.relu(tf.matmul(flat, W_3) + b_3)

# fully connected (10 hidden units)
W_4 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1))
b_4 = tf.Variable(tf.zeros([10]))
fc_2 = tf.matmul(fc_1, W_4) + b_4

# softmax and cross entropy (tf.nn.softmax_cross_entropy_with_logits is deprecated)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc_2, labels=y)
avg_loss = tf.reduce_mean(loss)

train_step = tf.train.AdamOptimizer(1e-4).minimize(avg_loss)


# In[4]:


sess = tf.Session()
# sess.run(tf.initialize_all_variables()) # deprecated
sess.run(tf.global_variables_initializer())

# train in multiple (10000) "epochs"
# really ~18 epochs in the D1 MLP notation with 550 mini batches of 100 images
# for simplicity just do both loops in one for loop though
for _ in trange(10000):
    # sweep through all the training data in mini batchs of 100 images
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})


# ### Evaluate model

# In[5]:


print('test avg loss = {0:.2f}'.format(sess.run(avg_loss, feed_dict={x:mnist.test.images, y:mnist.test.labels})))

y_test_pred = sess.run(fc_2, feed_dict={x:mnist.test.images, y:mnist.test.labels})
ncorrect = np.sum(np.argmax(y_test_pred, axis=1) == np.argmax(mnist.test.labels, axis=1))
print('test accuracy = {0:.2f}%'.format(100*(float(ncorrect) / float(mnist.test.labels.shape[0]))))


# ### Short answer
# 
# 1\. How does the CNN compare in accuracy with yesterday's logistic regression and MLP models? How about training time?

# Yesterday's MLP took 01:53 for 50 epochs and had a test accuracy of 97.22%  
# 
# Today's CNN took 32:23 for 10k "epochs" and had a test accuracy of 99.11%  
# That is a slight improvment at the cost of extra training, but in practice we could have just started with a different network pretrained on MNIST by someone else

# 2\. How many trainable parameters are there in the CNN you built for this assignment?
# 
# *Note: By trainable parameters, I mean individual scalars. For example, a weight matrix that is 10x5 has 50.*

# In[6]:


(5*5*1*32)+32
+(5*5*32*64)+64
+(7*7*64*256)+256
+(256*10)+10


# 3\. When would you use a CNN versus a logistic regression model or an MLP?

# CNN: Image classification, very high dimensionality inputs where each variable should be processed in relation to it's neighbors  
# Logistic regression / MLP: moderate dimensionality inputs, variables are not closely related
