from __future__ import print_function

import tensorflow as tf

#Paramètre
learning_rate = 0.1 	#indice d'apprentissage
num_steps = 1000    	#nombre d'étape
batch_size = 10		#taille des batch - inutile
display_step = 100	#fréquence d'affichage

#Paramètre du réseau
n_hidden_1 = 64		#nombre de neurones couche 1
n_hidden_2 = 64		#nombre de neurones couche 2
num_input = 32		#nombre d'entrée, à changer en fonction du réseau 
num_classes = 2		#nombre de sortie attendue toujours 2 => lors de la création du batch de sortie, prévoir 2 colonnes 1 pour chaque sortie

#tf Graph input
X = tf.placeholder("float",[None,num_input])		#configuration de l'ensemble des entrées
Y = tf.placeholder("float",[None,num_classes])		#configuration de l'ensemble des sorties

#Store layers weight et bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input,n_hidden_1])),	#initialisation des poids entrées - couche 1
    'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),	#initialisation des poids couche 1 - couche 2
    'out': tf.Variable(tf.random_normal([n_hidden_2,num_classes]))	#initialisation des poids couche 2 - sortie
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),			#initialisation des biais couche 1
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),			#initialisation des biais couche 2
    'out': tf.Variable(tf.random_normal([num_classes]))			#initialisation des biais sortie
}

#Create model
def neural_net(x):
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])		#calcul premiere couche
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])	#calcul seconde couche
    out_layer = tf.matmul(layer_2,weights['out'])+biases['out']		#calcul sortie
    return out_layer


#construct modelœ
logits = neural_net(X)							#création réseau de neurones
prediction = tf.nn.softmax(logits)					#lissage de la sortie

#define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))	#calcul de l'erreur
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)			#initialisation descente du gradient
train_op = optimizer.minimize(loss_op)								#optimisation

#evaluate model
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#initialize the variable
init = tf.global_variables_initializer()				#initialisation des variables globales

#start training
with tf.Session() as sess:
    sess.run(init)							#run de l'init

    for step in range(1, num_steps+1):					#boucle d'apprentissage
        batch_x, batch_y = #						#récupération des données

        sess.run(train_op, feed_dict={X: batch_x, Y:batch_y})		#lancement de l'apprentissage
        if step % display_step == 0 or step = 1:			#affichage
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X:batch_x, Y: batch_y}) 	#calcul de l'erreur
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
