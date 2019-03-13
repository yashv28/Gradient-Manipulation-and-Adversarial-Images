import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image
import random


max_iters = 6000
batch_size = 1
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

x2 = tf.placeholder(tf.float32, [None, 28, 28, 1])
y2 = tf.placeholder(tf.int32, [None, 10])


def network(x, batch_size):

	with tf.variable_scope('infer') as scope:

		# conv1
		conv1 = tf.layers.conv2d(inputs=x,filters=32,kernel_size=[3,3],padding='SAME',trainable=False,activation=tf.nn.relu,name='conv1')

		# conv2
		conv2 = tf.layers.conv2d(inputs=conv1,filters=64,kernel_size=[3,3],padding='SAME',trainable=False,activation=tf.nn.relu,strides=2,name='conv2')

		# conv3
		conv3 = tf.layers.conv2d(inputs=conv2,filters=128,kernel_size=[3,3],padding='SAME',trainable=False,activation=tf.nn.relu,strides=2,name='conv3')

		# conv4
		conv4 = tf.layers.conv2d(inputs=conv3,filters=128,kernel_size=[3,3],padding='SAME',trainable=False,activation=tf.nn.relu,strides=2,name='conv4')

		# conv5
		conv5 = tf.layers.conv2d(inputs=conv4,filters=128,kernel_size=[3,3],padding='SAME',trainable=False,activation=tf.nn.relu,strides=2,name='conv5')

		# fc4
		fc1 = tf.layers.dense(tf.reshape(conv5, [-1, conv5.shape[1]*conv5.shape[2]*conv5.shape[3]]),100,trainable=False,activation=tf.nn.relu,name='fc1')

		# local5
		fc2 = tf.layers.dense(fc1,10,trainable=False,name='fc2')

	return fc2

@tf.RegisterGradient("MySingGrad1")
def custom_grad(unused_op, dy):
	grad = tf.sign(dy)
	return tf.sign(tf.cast(grad < 0, dy.dtype) * dy)

Ie = tf.get_variable("Ie", [1, 28, 28, 1], initializer=tf.zeros_initializer)
g = tf.get_default_graph()

with g.gradient_override_map({"Identity" : "MySingGrad1"}):
	Ie_act = tf.identity(Ie, name="Identity")

I_pert = x2 + Ie_act

i_pert = tf.clip_by_value(I_pert, -1, 1)

x_tr = network(i_pert, batch_size)

sf = tf.nn.softmax(x_tr)

ypred = tf.cast(tf.argmax(sf, axis=1), tf.int32)
ytr = tf.cast(tf.argmax(y2, axis=1), tf.int32)
accuracy_te = tf.reduce_mean(tf.cast(tf.equal(ypred, ytr), tf.float32))

loss_adv = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x_tr, labels=y2))
optim_step = tf.train.AdamOptimizer(0.01).minimize(loss_adv)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
img = nxtest[4]
img = img[np.newaxis,:,:,:]
a = sess.run([x_tr], feed_dict={x2: img})

saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='infer'))
saver.restore(sess, "sign/model.ckpt")

accuracies = []

num = []

t_acc = 0.0
for i in range(0, nxtest.shape[0], 100):
	nxte, nyte = nxtest[i:i+100], nytest[i:i+100]
	# print(i)
	test_accuracy, yp, yt = sess.run([accuracy_te, ypred, ytr], feed_dict={x2: nxte, y2: nyte})  
	if((i==5500) and (yp[55]==yt[55])):
		num.append(5555)
	if((i==2800) and (yp[12]==yt[12])):
		num.append(2812)
	if((i==1500) and (yp[90]==yt[90])):
		num.append(1590)
	if((i==9400) and (yp[12]==yt[12])):
		num.append(9412)
	t_acc += float(test_accuracy * 100)

acc = t_acc/float(nxtest.shape[0])
print("Test accuracy:", acc)
print(len(num))

demo_steps = 200
rn = random.choice(num)
imy = nytest[rn]
demo_target = imy[np.newaxis, :]
print(demo_target)
img = nxtest[rn]
np.save("./3.1_orig.npy",img)
img = img[np.newaxis,:,:,:]


for i in range(demo_steps):

	_, loss_value, yo, prob, adv = sess.run([optim_step, loss_adv, ypred, sf, i_pert], feed_dict={x2: img, y2: demo_target})
	conf = prob[0][yo[0]]
	if((conf>0.9) and (np.argmax(prob[0])!=np.argmax(imy))):
		break
	print("step",i+1,"loss=",loss_value,"y=",yo[0],"conf=",conf)

np.save("./3.1_final.npy",adv)
