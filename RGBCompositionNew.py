import torch
import torch.nn as nn
import torch.optim as optim
from skimage import io
import utils.utils as utils
import cv2
import numpy as np

# r_ori = cv2.flip(cv2.imread('./recon/ExpColor/r1.jpg'),1)
# g_ori = cv2.flip(cv2.imread('./recon/ExpColor/g1.jpg'),1)
# b_ori = cv2.flip(cv2.imread('./recon/ExpColor/b1.jpg'),1)
r_ori = cv2.imread('./recon/ExpColor/r1.jpg')
g_ori = cv2.imread('./recon/ExpColor/g1.jpg')
b_ori = cv2.imread('./recon/ExpColor/b1.jpg')
r_ori=r_ori[2035:3323,2962:5385,:]
g_ori=g_ori[2035:3323,2962:5385,:]
b_ori=b_ori[2035:3323,2962:5385,:]
# r_com=r_ori
# g_com, _, _ = utils.siftImageAlignment(r_ori, g_ori)
# b_com, _, _ = utils.siftImageAlignment(r_ori, b_ori)
g_com=g_ori
r_com, _, _ = utils.siftImageAlignment(g_ori, r_ori)
b_com, _, _ = utils.siftImageAlignment(g_ori, b_ori)
# b_com=b_ori
# r_com, _, _ = utils.siftImageAlignment(b_ori, r_ori)
# g_com, _, _ = utils.siftImageAlignment(b_ori, g_ori)
cv2.namedWindow('r_ori', cv2.WINDOW_NORMAL)
cv2.namedWindow('g_ori', cv2.WINDOW_NORMAL)
cv2.namedWindow('b_ori', cv2.WINDOW_NORMAL)
cv2.namedWindow('r_com', cv2.WINDOW_NORMAL)
cv2.namedWindow('g_com', cv2.WINDOW_NORMAL)
cv2.namedWindow('b_com', cv2.WINDOW_NORMAL)
cv2.imshow('r_ori', r_ori)
cv2.imshow('g_ori', g_ori)
cv2.imshow('b_ori', b_ori)
cv2.imshow('r_com', r_com)
cv2.imshow('g_com', g_com)
cv2.imshow('b_com', b_com)
cv2.waitKey(0)
# roi=[897,3011,794,4199]
# roi=[838,3014,935,4488]
# roi=[507,2114,530,3106]
roi=[450,2169,414,3165]
r_col=r_com[roi[0]:roi[1],roi[2]:roi[3],:]
g_col=g_com[roi[0]:roi[1],roi[2]:roi[3],:]
b_col=b_com[roi[0]:roi[1],roi[2]:roi[3],:]
r_col=cv2.resize(r_col,(2560,1600))
g_col=cv2.resize(g_col,(2560,1600))
b_col=cv2.resize(b_col,(2560,1600))
r = cv2.cvtColor(r_col, cv2.COLOR_BGR2RGB)
g = cv2.cvtColor(g_col, cv2.COLOR_BGR2RGB)
b = cv2.cvtColor(b_col, cv2.COLOR_BGR2RGB)

device = torch.device('cuda')
dtype = torch.float32
loss = nn.MSELoss().to(device)
slm_res = (1600,2560)
lr_s_rgb=4e-1
lr_offset_rgb=4e-1
lr_gamma_rgb=1e-1
num_iters=500

target_amp=torch.tensor(io.imread('./recon/ExpColor/ori.jpg') / 255, dtype=dtype).to(device)
target_amp_r=target_amp[:,:,0]
target_amp_g=target_amp[:,:,1]
target_amp_b=target_amp[:,:,2]
img_r=torch.tensor(r/255, dtype=dtype).to(device)
img_g=torch.tensor(g/255, dtype=dtype).to(device)
img_b=torch.tensor(b/255, dtype=dtype).to(device)

timeN=10
lossValueMin=torch.tensor(1.0, dtype=dtype).to(device)
img_r_s_min=torch.zeros_like(img_r)
img_g_s_min=torch.zeros_like(img_g)
img_b_s_min=torch.zeros_like(img_b)
for timen in range(timeN):
    # s0_rgb=[0.5,0.5,0.5]
    s0_rgb=np.random.rand(9)
    s_rgb = torch.tensor(s0_rgb, requires_grad=True, device=device)
    # # offset0_rgb=[0,0,0]
    # offset0_rgb=np.random.rand(9)
    # offset_rgb = torch.tensor(offset0_rgb, requires_grad=True, device=device)
    # gamma0_rgb = [1.,1.,1.,1.,1.,1.,1.,1.,1.]
    # gamma_rgb = torch.tensor(gamma0_rgb, requires_grad=True, device=device)

    optvars = [{'params': s_rgb, 'lr': lr_s_rgb}]
    # optvars = [{'params': s_rgb, 'lr': lr_s_rgb},{'params': offset_rgb, 'lr': lr_offset_rgb}]
    # optvars = [{'params': s_rgb, 'lr': lr_s_rgb},{'params': offset_rgb, 'lr': lr_offset_rgb},{'params': gamma_rgb, 'lr': lr_gamma_rgb}]
    optimizer = optim.Adam(optvars)
    for k in range(num_iters):
        # print(k)
        optimizer.zero_grad()

        img_r_s = img_r[:,:,0] * s_rgb[0]+img_g[:,:,0] * s_rgb[1]+img_b[:,:,0] * s_rgb[2]
        img_g_s = img_r[:,:,1] * s_rgb[3]+img_g[:,:,1] * s_rgb[4]+img_b[:,:,1] * s_rgb[5]
        img_b_s = img_r[:,:,2] * s_rgb[6]+img_g[:,:,2] * s_rgb[7]+img_b[:,:,2] * s_rgb[8]
        # img_r_s = img_r[:, :, 0] * s_rgb[0] + img_g[:, :, 0] * s_rgb[1] + img_b[:, :, 0] * s_rgb[2]+offset_rgb[0]+offset_rgb[1]+offset_rgb[2]
        # img_g_s = img_r[:, :, 1] * s_rgb[3] + img_g[:, :, 1] * s_rgb[4] + img_b[:, :, 1] * s_rgb[5]+offset_rgb[3]+offset_rgb[4]+offset_rgb[5]
        # img_b_s = img_r[:, :, 2] * s_rgb[6] + img_g[:, :, 2] * s_rgb[7] + img_b[:, :, 2] * s_rgb[8]+offset_rgb[6]+offset_rgb[7]+offset_rgb[8]
        # img_r_s = img_r[:, :, 0]**gamma_rgb[0] * s_rgb[0] + img_g[:, :, 0]**gamma_rgb[1] * s_rgb[1] + img_b[:, :, 0]**gamma_rgb[2] * s_rgb[2] + offset_rgb[0] + offset_rgb[1] + offset_rgb[2]
        # img_g_s = img_r[:, :, 1]**gamma_rgb[3] * s_rgb[3] + img_g[:, :, 1]**gamma_rgb[4] * s_rgb[4] + img_b[:, :, 1]**gamma_rgb[5] * s_rgb[5] + offset_rgb[3] + offset_rgb[4] + offset_rgb[5]
        # img_b_s = img_r[:, :, 2]**gamma_rgb[6] * s_rgb[6] + img_g[:, :, 2]**gamma_rgb[7] * s_rgb[7] + img_b[:, :, 2]**gamma_rgb[8] * s_rgb[8] + offset_rgb[6] + offset_rgb[7] + offset_rgb[8]
        img_max=utils.max([img_r_s.max(),img_g_s.max(),img_b_s.max()])
        img_r_s=img_r_s/img_max
        img_g_s = img_g_s / img_max
        img_b_s = img_b_s / img_max

        mse_loss_r = loss(img_r_s, target_amp_r)
        mse_loss_g = loss(img_g_s, target_amp_g)
        mse_loss_b = loss(img_b_s, target_amp_b)
        lossValue=mse_loss_r+mse_loss_g+mse_loss_b
        lossValue.backward()
        optimizer.step()
    print('lossValue:', lossValue)
    print('s_rgb0:',s0_rgb)
    print('s_rgb:', s_rgb.cpu().detach().numpy())
    # print('offset0_rgb:',offset0_rgb)
    # print('offset_rgb:', offset_rgb.cpu().detach().numpy())
    # print('gamma0_rgb:', gamma0_rgb)
    # print('gamma_rgb:', gamma_rgb.cpu().detach().numpy())
    print('*********************')

    if lossValue<=lossValueMin:
        lossValueMin=lossValue
        img_r_s_min=img_r_s
        img_g_s_min = img_g_s
        img_b_s_min = img_b_s
print('lossValueMin:', lossValueMin)
out_amp=torch.zeros_like(target_amp)
out_amp[:,:,0]=img_r_s_min
out_amp[:, :, 1] = img_g_s_min
out_amp[:, :, 2] = img_b_s_min
cv2.namedWindow('out_amp', cv2.WINDOW_NORMAL)
cv2.namedWindow('target_amp', cv2.WINDOW_NORMAL)
cv2.imshow('out_amp', cv2.cvtColor(out_amp.cpu().detach().numpy(), cv2.COLOR_RGB2BGR))
cv2.imshow('target_amp', cv2.cvtColor(target_amp.cpu().detach().numpy(), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)