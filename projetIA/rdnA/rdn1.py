from __future__ import print_function

import tensorflow as tf
import extractor as ext

learning_rate = 0.01
training_epochs = 25
batch_size = 150
display_epoch = 1
logs_path = '/tmp/tensorflow_logs/rdn/'

x = tf.placeholder(tf.float32, [None, 32], name = "InputData")
y = tf.placeholder(tf.float32, [None, 2], name = "OutputData")

w = tf.Variable(tf.zeros([32, 2]), name='weights')
b = tf.Variable(tf.zeros([2]), name='bias')

with tf.name_scope('Model'):
	pred = tf.nn.softmax(tf.matmul(x,w) + b)

with tf.name_scope('Loss'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    cost = -tf.reduce_sum(y*tf.log(tf.clip_by_value(pred,1e-10,1.0)))
	 #cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))


with tf.name_scope('SGD'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
	acc = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
	acc = tf.reduce_mean(tf.cast(acc,tf.float32))
    
tf.summary.scalar("loss",cost)
tf.summary.scalar("accuracy",acc)
merged_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
#start training
with tf.Session() as sess:
    sess.run(init)							#run de linit
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    ll = ext.extractor_1("wpbc.data")
    l = ext.extractor_utils(ll)
    lll = ext.extractor_res(ll)
    a = ext.list_string_to_int(lll)
    le = ext.list_list_string_to_int(l)
    #le, a = ext.upgrade_list(le, a, 1000)
    b = ext.double_sortie(a)
    le = tf.nn.l2_normalize(le, 1, epsilon=1e-12)
    le = le.eval()
    batch_x = le[:batch_size]
    test_x = le[batch_size:]
    batch_y = b[:batch_size]
    test_y = b[batch_size:]
    for epoch in range(training_epochs):
        avg_cost = 0.

        _, c, summary = sess.run([optimizer, cost, merged_summary_op],feed_dict={x:batch_x , y:batch_y})
        summary_writer.add_summary(summary, epoch * batch_size)
        avg_cost += c / batch_size
        if(epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost= %s" % avg_cost)
    print("Accuracy:", acc.eval({x:test_x , y:test_y}))
