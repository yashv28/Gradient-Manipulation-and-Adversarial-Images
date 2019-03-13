import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

accr = np.load("2.1_relu.npy")
accs = np.load("2.3_sign.npy")
accrt = np.load("2.1_relu_test.npy")
accst = np.load("2.3_sign_test.npy")

plt.plot(np.arange(len(accr)), accr)
plt.xlabel("#Iterations")
plt.ylabel("Training Accuracy")
plt.show()
plt.waitforbuttonpress()
plt.close()

plt.plot(np.arange(len(accs)), accs)
plt.xlabel("#Iterations")
plt.ylabel("Training Accuracy")
plt.show()
plt.waitforbuttonpress()
plt.close()

plt.plot(np.arange(len(accrt)), accrt)
plt.xlabel("#Iterations")
plt.ylabel("Testing Accuracy")
plt.show()
plt.waitforbuttonpress()
plt.close()

plt.plot(np.arange(len(accst)), accst)
plt.xlabel("#Iterations")
plt.ylabel("Testing Accuracy")
plt.show()
plt.waitforbuttonpress()
plt.close()
