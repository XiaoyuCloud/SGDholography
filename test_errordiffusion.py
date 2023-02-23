import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils.utils as utils

#input
im1 = cv2.imread('./data/1.png')
im1=im1.astype(np.double)

#rgb2gray
im1=(im1[:,:,0]+im1[:,:,1]+im1[:,:,2])/3
im1=utils.fieldNormal(im1)
plt.figure('im1')
plt.imshow(im1,cmap='gray')
# plt.show()

type='Jarvis-Jidice-Ninke'
im2=utils.ErrorDiffusion(im1,type,dtype='float32',numten='numpy',thresh=0.5,min=0.,max=1.)
cv2.imwrite('ed1.png',im2)
plt.figure('im2')
plt.imshow(im2,cmap='gray')
plt.show()
