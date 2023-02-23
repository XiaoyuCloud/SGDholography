import numpy as np
import matplotlib.pyplot as plt
import utils.utils as utils

# parameters for base
slm_res = (600,600)
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
feature_size = (8*1 * um, 8*1 * um)
p=1
l = 1
w = 0.5e-3

# instantiation and holo calc
Field = utils.LGmode(slm_res, feature_size,p,l, w)
plt.figure('f1')
plt.imshow(np.angle(Field), interpolation='none',cmap='jet')
plt.figure('f2')
plt.imshow(abs(Field), interpolation='none',cmap='jet')
plt.show()