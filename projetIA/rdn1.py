from __future__ import print_function
import tensorflow as tf
import extractor as ext

learning_rate = 0.01
training_epochs = 25
batch_size = 150
display_epoch = 1
logs_path = '/tmp/tensorflow_logs/rdn/'

x = tf.placeholder(tf.float32, [None, 32], name='InputData')
y = tf.placeholder(tf.float32, [None, 2], name='LabelData')

w = tf.Variable(tf.zeros([32,2]), name='weights')
b = tf.Variable(tf.zeros([2]), name='bias')

with tf.name_scope('Model'):
	pred = tf.nn.relu(tf.matmul(x,w) + b)
	print(pred)

with tf.name_scope('Loss'):
	cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

with tf.name_scope('SGD'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)		

with tf.name_scope('Accuracy'):
	acc = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
	acc = tf.reduce_mean(tf.cast(acc,tf.float32))

init = tf.global_variables_initializer()

tf.summary.scalar("loss",cost)
tf.summary.scalar("accuracy",acc)
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:

	sess.run(init)
	summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	
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
		#batch
		
		_, c, summary = sess.run([optimizer, cost, merged_summary_op],feed_dict={x:batch_x , y:batch_y})
		print("\n c : ")
		print(c)
		print("\n")
		summary_writer.add_summary(summary, epoch * 150)
		avg_cost += c / 150
		if(epoch+1) % display_epoch == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}", format(avg_cost))
	
	print("Optimization finished")

	
	print("Accuracy:", acc.eval({x:test_x , y:test_y}))
	print("Run the command line:\n" \
		"--> tensorboard --logdir=/tmp/tensorflow_logs " \
"\nThen open http://0.0.0.0:6006/ into your web browser")
