
# coding: utf-8

# In[1]:

import tensorflow.python.platform
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)
    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


# In[2]:

import tensorflow as tf
sess = tf.InteractiveSession()
print tf.__version__
x = tf.placeholder("float", shape=[None, 784], name='x-input')
y_ = tf.placeholder("float", shape=[None, 10], name='y-input')
W_conv1 = weight_variable([5, 5, 1, 32],'Weight_conv1')
b_conv1 = bias_variable([32], 'bias_conv1')
with tf.name_scope('h_conv1'):
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
with tf.name_scope('h_pool1'):    
    h_pool1 = max_pool_2x2(h_conv1)


# In[3]:


W_conv2 = weight_variable([5, 5, 32, 64],'Weight_conv2')
b_conv2 = bias_variable([64], 'bias_conv2')
with tf.name_scope('h_conv2'):
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
with tf.name_scope('h_pool2'):
    h_pool2 = max_pool_2x2(h_conv2)


# In[4]:

W_fc1 = weight_variable([7 * 7 * 64, 512],'Weight_fc1')
b_fc1 = bias_variable([512],'bias_fc1')
with tf.name_scope('h_pool2_flat'):
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
with tf.name_scope('h_fc1'):
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
with tf.name_scope('h_fc1_drop'):
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
W_fc2 = weight_variable([512, 10],'Weight_fc2')
b_fc2 = bias_variable([10],'bias_fc2')

with tf.name_scope('y'):
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

_ = tf.histogram_summary('Weight_conv1', W_conv1)
_ = tf.histogram_summary('bias_conv1', b_conv1)
_ = tf.histogram_summary('Weight_conv2', W_conv2)
_ = tf.histogram_summary('bias_conv2', b_conv2)
_ = tf.histogram_summary('Weight_fc1', W_fc1)
_ = tf.histogram_summary('bias_fc1', b_fc1)
_ = tf.histogram_summary('Weight_fc2', W_fc2)
_ = tf.histogram_summary('bias_fc2', b_fc2)
_ = tf.histogram_summary('y', y_conv)


# In[ ]:

with tf.name_scope('xent'):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))  
    _ = tf.scalar_summary('cross entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
with tf.name_scope('test'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    _ = tf.scalar_summary('accuracy', accuracy)

# Merge all the summaries and write them out to /tmp/mnist_logs
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter('/tmp/mnist_logs', sess.graph_def)    
tf.initialize_all_variables().run()

for i in range(10000):
    batch_data_train = mnist.train.next_batch(100)
    if i% 10 == 0:
        feed_dict_train={x:batch_data_train[0], y_: batch_data_train[1], keep_prob: 1.0}
        feed_dict_test={x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
        summary_str,accuracy_train=sess.run( [merged,accuracy], feed_dict=feed_dict_train)
        writer.add_summary(summary_str, i)
        accuracy_test=sess.run( accuracy, feed_dict=feed_dict_test)
        print('Accuracy train at step %s: %s' % (i, accuracy_train))
        print('         test at step %s: %s' % (i, accuracy_test))
    feed_dict_train={x:batch_data_train[0], y_: batch_data_train[1], keep_prob: 0.5}
    sess.run(train_step, feed_dict=feed_dict_train)

    
#print("test accuracy %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



