
# coding: utf-8

# # TensorFlow Assignment: Multi-Layer Perceptron (MLP)

# **[Duke Community Standard](http://integrity.duke.edu/standard.html): By typing your name below, you are certifying that you have adhered to the Duke Community Standard in completing this assignment.**
# 
# Name: Matthew Epland

# ### Multi-layer Perceptron
# 
# Build a 2-layer MLP for MNIST digit classfication. Feel free to play around with the model architecture and see how the training time/performance changes, but to begin, try the following:
# 
# Image (784 dimensions) -> fully connected layer (500 hidden units) -> nonlinearity (ReLU) -> fully connected layer (100 hidden units) -> nonlinearity (ReLU) -> fully connected (10 hidden units) -> softmax

# ### Imports

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tqdm import trange


# In[2]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);


# ### Setup the tf model

# Image (784 dimensions) -> fully connected layer (500 hidden units) -> nonlinearity (ReLU) -> fully connected layer (100 hidden units) -> nonlinearity (ReLU) -> fully connected (10 hidden units) -> softmax

# In[3]:


tf.reset_default_graph()

# placeholders are input nodes in the graph, take in data from train/test set
# are variable so we can pick our own mini batch size and feed it in
X = tf.placeholder(tf.float32, [None,784]) # Image (784 dimensions)
y = tf.placeholder(tf.float32, [None,10])

# fully connected layer (500 hidden units) -> nonlinearity (ReLU)
W_0 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b_0 = tf.Variable(tf.truncated_normal([500], stddev=0.1))
scores_0 = tf.nn.relu(tf.matmul(X, W_0) + b_0)

# fully connected layer (100 hidden units) -> nonlinearity (ReLU)
W_1 = tf.Variable(tf.truncated_normal([500, 100], stddev=0.1))
b_1 = tf.Variable(tf.truncated_normal([100], stddev=0.1))
scores_1 = tf.nn.relu(tf.matmul(scores_0, W_1) + b_1)

# fully connected layer (10 hidden units) -> nonlinearity (ReLU)
W_2 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
b_2 = tf.Variable(tf.truncated_normal([10], stddev=0.1))
scores = tf.matmul(scores_1, W_2) + b_2

# softmax and cross entropy (tf.nn.softmax_cross_entropy_with_logits is deprecated)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores, labels=y)
avg_loss = tf.reduce_mean(loss)

# learning rate is 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(avg_loss)


# ### Train model

# In[4]:


sess = tf.Session()
# sess.run(tf.initialize_all_variables()) # deprecated
sess.run(tf.global_variables_initializer())

# train in multiple epochs (50)
for epoch in trange(50):
    # sweep through all the training data in mini batchs of 100 images
    for i in range(550):
        start = i*100
        end = (i+1)*100

        # do one step of gradient descent
        sess.run(train_step, feed_dict={X:mnist.train.images[start:end],
                                        y:mnist.train.labels[start:end]})


# ### Evaluate model

# In[5]:


print('test avg loss = {0:.2f}'.format(sess.run(avg_loss, feed_dict={X:mnist.test.images, y:mnist.test.labels})))

y_test_pred = sess.run(scores, feed_dict={X:mnist.test.images, y:mnist.test.labels})
ncorrect = np.sum(np.argmax(y_test_pred, axis=1) == np.argmax(mnist.test.labels, axis=1))
print('test accuracy = {0:.2f}%'.format(100*(float(ncorrect) / float(mnist.test.labels.shape[0]))))

