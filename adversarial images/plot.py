import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random

names = ["3.3_final.npy", "3.3_orig.npy", "3.1_final.npy", "3.1_orig.npy"]

for n in names:
	a = np.load(n)
	h = a+1.0
	i = h/2.0
	j = i*255.0
	j = j.reshape(28,28).astype('uint8')
	plt.imshow(j, cmap='gray')
	plt.show()
