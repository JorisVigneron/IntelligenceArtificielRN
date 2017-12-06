from __future__ import print_function

import tensorflow as tf
import extractor as ext

#Parametre
learning_rate = 0.1 	#indice d'apprentissage
num_steps = 100   	#nombre d'etape
batch_size = 10		#taille des batch - inutile
display_step = 10	#frequence daffichage

#Parametre du reseau
n_hidden_1 = 64		#nombre de neurones couche 1
n_hidden_2 = 64		#nombre de neurones couche 2
num_input = 32		#nombre dentree a changer en fonction du reseau 
num_classes = 2		#nombre de sortie attendue toujours 2 => lors de la creation du batch de sortie, prevoir 2 colonnes 1 pour chaque sortie

#tf Graph input
X = tf.placeholder("float",[None,num_input])		#configuration de lensemble des entrees
Y = tf.placeholder("float",[None,num_classes])		#configuration de lensemble des sorties

#Store layers weight et bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input,n_hidden_1])),	#initialisation des poids entrees - couche 1
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


#construct modele
logits = neural_net(X)							#creation reseau de neurones
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
    print(len(batch_x))
    print(len(batch_y))
    for step in range(1, num_steps+1):					#boucle dapprentissage
         					#recuperation des donnees

        sess.run(train_op, feed_dict={X: batch_x, Y:batch_y})		#lancement de lapprentissage
        if step % display_step == 0 or step == 1:			#affichage
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X:batch_x, Y: batch_y}) 	#calcul de lerreur
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
