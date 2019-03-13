import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def weight_variable(shape, mean=0, stddev=0.1):
	# ini = tf.initializers.variance_scaling()
	initialization = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
	return tf.Variable(initialization)


def bias_variable(shape, start_val=0.1):
	initialization = tf.constant(start_val, shape=shape)
	return tf.Variable(initialization)


max_iters = 10000
batch_size = 100
learning_rate = 3e-4
normalize = False
flip = False
pad_n_crop = False

(nxtrain, nytrain), (nxtest, nytest) = tf.keras.datasets.cifar10.load_data()

nxtrain = nxtrain.astype(np.float32)
nxtest = nxtest.astype(np.float32)
nytrain = np.eye(np.max(nytrain)+1)[nytrain.flatten()]
nytest = np.eye(np.max(nytest)+1)[nytest.flatten()]

tf.reset_default_graph()

weights = {
	'w1': weight_variable([5,5,3,32]),
	'w2': weight_variable([5,5,32,32]),
	'w3': weight_variable([5,5,32,64]),
	'fc1': weight_variable([4*4*64,64]),
	'fc2': weight_variable([64,10])
}

biases = {
	'b1': bias_variable([1,1,1,32]),
	'b2': bias_variable([1,1,1,32]),
	'b3': bias_variable([1,1,1,64]),
	'fb1': bias_variable([1,64]),
	'fb2': bias_variable([1,10])
}

x1 = tf.placeholder(tf.float32, [None, 32, 32, 3])
y1 = tf.placeholder(tf.int32, [None, 10])
x2 = tf.placeholder(tf.float32, [None, 32, 32, 3])
y2 = tf.placeholder(tf.int32, [None, 10])


def network(x, labels, batch_size, train):

	if(normalize):
		x = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x)
		z = x
	if(train):
		if(flip):
			x = tf.image.random_flip_left_right(x)
		if(pad_n_crop):
			x = tf.pad(x, [[0,0],[4,4],[4,4],[0,0]])
			x = tf.map_fn(lambda frame: tf.random_crop(frame, [32,32,3]), x)

	# conv1
	x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME') + biases['b1']
	x = tf.contrib.layers.batch_norm(x)
	x = tf.nn.relu(x)
	x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

	# conv2
	x = tf.nn.conv2d(x, weights['w2'], strides=[1,1,1,1], padding='SAME') + biases['b2']
	x = tf.contrib.layers.batch_norm(x)
	x = tf.nn.relu(x)
	x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

	# conv3
	x = tf.nn.conv2d(x, weights['w3'], strides=[1,1,1,1], padding='SAME') + biases['b3']
	x = tf.contrib.layers.batch_norm(x)
	x = tf.nn.relu(x)
	x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

	# fc4
	x = tf.reshape(x, [batch_size, -1])
	x = tf.matmul(x, weights['fc1']) + biases['fb1']
	x = tf.contrib.layers.batch_norm(x)
	x = tf.nn.relu(x)

	# local5
	x = tf.matmul(x, weights['fc2']) + biases['fb2']

	# Softmax
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=labels))
	y_hat = tf.cast(tf.argmax(tf.nn.softmax(x), axis=1), tf.int32)
	y_true = tf.cast(tf.argmax(labels, axis=1), tf.int32)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(y_hat, y_true), tf.float32))

	return loss, accuracy, z


tf_loss, accuracy_tr, z = network(x1, y1, batch_size, True)
_, accuracy_te, _ = network(x2, y2, batch_size, False)

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(tf_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses = []
accuracies = []

max_epochs = 20

accuracies = []

for epoch in range(max_epochs):

	for i in range(0, nxtrain.shape[0], batch_size):

		nxtr, nytr = nxtrain[i:i+batch_size], nytrain[i:i+batch_size]
		_, loss, accuracy, s = sess.run([train, tf_loss, accuracy_tr, z], feed_dict={x1: nxtr, y1: nytr})
		if(i%100==0):
			accuracies.append(accuracy)
		if(i%50000==0):
			print(s)
			print("Iteration:", int(epoch*(max_iters/max_epochs) + (i/batch_size)), "Accuracy:", accuracy, "Loss:", loss.mean())

t_acc = 0.0
for i in range(0, nxtest.shape[0], batch_size):
	nxte, nyte = nxtest[i:i+batch_size], nytest[i:i+batch_size]

	test_accuracy = sess.run([accuracy_te], feed_dict={x2: nxte, y2: nyte})  
	t_acc += float(test_accuracy[0] * batch_size)

acc = t_acc/float(nxtest.shape[0])
print("Test accuracy:", acc)

if(pad_n_crop):
	np.save("./yash/pad_n_crop.npy",accuracies)
if(flip):
	if(pad_n_crop==False):
		np.save("./yash/flip.npy",accuracies)
if(normalize):
	if(flip==False):
		np.save("./yash/normalize.npy",accuracies)
if(normalize==False):
	np.save("./yash/raw.npy",accuracies)
