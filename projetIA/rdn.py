from __future__ import print_function

import tensorflow as tf
import extractor as ext

learning_rate = 0.01
training_epochs = 25
batch_size = 150
display_epoch = 1
logs_path = '/tmp/tensorflow_logs/rdn/'

x = tf.placeholder(tf.float32, [None, 32])
y = tf.placeholder(tf.float32, [None, 2])

w = tf.Variable(tf.zeros([32,2]), name='weights')
b = tf.Variable(tf.zeros([2]), name='bias')

with tf.name_scope('Model'):
	pred = tf.nn.softmax(tf.matmul(x,w) + b)

with tf.name_scope('Loss'):
	cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

with tf.name_scope('SGD'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
	acc = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
	acc = tf.reduce_mean(tf.cast(acc,tf.float32))

init = tf.global_variables_initializer()
#start training
with tf.Session() as sess:
    sess.run(init)							#run de linit
    ll = ext.extractor_1("wpbc.data")
    l = ext.extractor_utils(ll)
    lll = ext.extractor_res(ll)
    a = ext.list_string_to_int(lll)
    b = ext.double_sortie(a)
    batch_x = l[:150]
    test_x = l[150:]
    batch_y = b[:150]
    test_y = b[150:]
    for epoch in range(training_epochs):
        avg_cost = 0.

        _, c = sess.run([optimizer, cost],feed_dict={x:batch_x , y:batch_y})
        avg_cost += c / 150
        if(epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}", format(avg_cost))
