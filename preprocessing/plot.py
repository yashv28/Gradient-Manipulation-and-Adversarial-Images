import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

accr = np.load("acc/raw.npy")
accn = np.load("acc/normalize.npy")
accf = np.load("acc/flip.npy")
accp = np.load("acc/pad_n_crop.npy")

t = []
for i in range(len(accr)):
	if(i%500==0):
		t.append(accr[i])
accr = t

t = []
for i in range(len(accn)):
	if(i%500==0):
		t.append(accn[i])
accn = t

t = []
for i in range(len(accf)):
	if(i%500==0):
		t.append(accf[i])
accf = t

t = []
for i in range(len(accp)):
	if(i%500==0):
		t.append(accp[i])
accp = t

plt.plot(np.arange(len(accr)), accr)
plt.xlabel("#Epochs")
plt.ylabel("Training Accuracy")
plt.show()
plt.waitforbuttonpress()
plt.close()

plt.plot(np.arange(len(accn)), accn)
plt.xlabel("#Epochs")
plt.ylabel("Training Accuracy")
plt.show()
plt.waitforbuttonpress()
plt.close()

plt.plot(np.arange(len(accf)), accf)
plt.xlabel("#Epochs")
plt.ylabel("Training Accuracy")
plt.show()
plt.waitforbuttonpress()
plt.close()

plt.plot(np.arange(len(accp)), accp)
plt.xlabel("#Epochs")
plt.ylabel("Training Accuracy")
plt.show()
plt.waitforbuttonpress()
plt.close()
