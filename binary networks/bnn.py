import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


max_iters = 6000
batch_size = 100
learning_rate = 3e-4

(nxtrain, nytrain), (nxtest, nytest) = tf.keras.datasets.mnist.load_data()

nxtrain = nxtrain.astype(np.float32)[:, :, :, np.newaxis]
nxtrain = (2.0*(nxtrain/255.0)) - 1.0
nxtest = nxtest.astype(np.float32)[:, :, :, np.newaxis]
nxtest = (2.0*(nxtest/255.0)) - 1.0
nytrain = np.eye(np.max(nytrain)+1)[nytrain.flatten()]
nytest = np.eye(np.max(nytest)+1)[nytest.flatten()]


tf.reset_default_graph()

@tf.custom_gradient
def my_sign(x):
	c = tf.sign(x)
	def grad(dy):
		return tf.multiply(tf.cast((tf.abs(x)<=1), tf.float32), dy)
	
	return c, grad


# Activation -> tf.nn.relu/my_sign
activation = my_sign  

x1 = tf.placeholder(tf.float32, [None, 28, 28, 1])
y1 = tf.placeholder(tf.int32, [None, 10])


def network(x, batch_size):

	# conv1
	conv1 = tf.layers.conv2d(inputs=x,filters=32,kernel_size=[3,3],padding='SAME',activation=activation,name='conv1')

	# conv2
	conv2 = tf.layers.conv2d(inputs=conv1,filters=64,kernel_size=[3,3],padding='SAME',activation=activation,strides=2,name='conv2')

	# conv3
	conv3 = tf.layers.conv2d(inputs=conv2,filters=128,kernel_size=[3,3],padding='SAME',activation=activation,strides=2,name='conv3')

	# conv4
	conv4 = tf.layers.conv2d(inputs=conv3,filters=128,kernel_size=[3,3],padding='SAME',activation=activation,strides=2,name='conv4')

	# conv5
	conv5 = tf.layers.conv2d(inputs=conv4,filters=128,kernel_size=[3,3],padding='SAME',activation=activation,strides=2,name='conv5')

	# fc4
	fc1 = tf.layers.dense(tf.reshape(conv5, [-1, conv5.shape[1]*conv5.shape[2]*conv5.shape[3]]),100,activation=activation,name='fc1')

	# local5
	fc2 = tf.layers.dense(fc1,10,name='fc2')

	return fc2

x_tr = network(x1, batch_size)

y_hat = tf.cast(tf.argmax(tf.nn.softmax(x_tr), axis=1), tf.int32)
y_true = tf.cast(tf.argmax(y1, axis=1), tf.int32)

tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x_tr, labels=y1))

accuracy_tr = tf.reduce_mean(tf.cast(tf.equal(y_hat, y_true), tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(tf_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses = []
accuracies = []

max_epochs = int(max_iters/600)

accuracies = []

for epoch in range(max_epochs):

	for i in range(0, nxtrain.shape[0], batch_size):

		nxtr, nytr = nxtrain[i:i+batch_size], nytrain[i:i+batch_size]

		_, loss, accuracy = sess.run([train, tf_loss, accuracy_tr], feed_dict={x1: nxtr, y1: nytr})
		if(i%60000==0):
			print("Iteration:", int(epoch*(max_iters/max_epochs) + (i/batch_size)), "Accuracy:", accuracy, "Loss:", loss.mean())
		if(i%100==0):
			accuracies.append(accuracy)

t_acc = 0.0
accl = []
for i in range(0, nxtest.shape[0], batch_size):
	nxte, nyte = nxtest[i:i+batch_size], nytest[i:i+batch_size]

	test_accuracy = sess.run([accuracy_tr], feed_dict={x1: nxte, y1: nyte})  
	t_acc += float(test_accuracy[0] * batch_size)
	accl.append(test_accuracy[0])

acc = t_acc/float(nxtest.shape[0])
print("Test accuracy:", acc)

if(activation==my_sign):
	np.save("./2.3_sign.npy",accuracies)
	np.save("./2.3_sign_test.npy",accl)
else:
	np.save("./2.1_relu.npy",accuracies)
	np.save("./2.1_relu_test.npy",accl)
