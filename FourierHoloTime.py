import torch
import torch.nn as nn
# from skimage import io
import scipy.io as io
import configargparse
import utils.utils as utils
import matplotlib.pyplot as pyplot
import os
import numpy as np
from propagation_ASM import propagation_ASM
import torch.optim as optim
from pytorch_msssim import ssim as ssim_func
import torch.nn.functional as F
import cv2
import torch.fft as tfft

p = configargparse.ArgumentParser()
p.add_argument('--channel', type=int, default=0, help='Red:0, green:1, blue:2')
p.add_argument('--root_path', type=str, default='./phases', help='Directory where optimized phases will be saved.')
p.add_argument('--prop_model', type=str, default='ASM',help='Type of propagation model for reconstruction: ASM / MODEL / CAMERA')
opt = p.parse_args()

channel = opt.channel
chan_str = ('red', 'green', 'blue')[channel]
root_path = os.path.join(opt.root_path, 'SGD_ASM', chan_str)
slm_res = (1600,2560)
roi_res = (1438,2300)
device = torch.device('cuda')
dtype = torch.float32
loss = nn.MSELoss().to(device)
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
feature_size = (7.56*1 * um, 7.56*1 * um)
wavelengths = (638 * nm, 520 * nm, 450 * nm)
z=20
prop_dists = (z*cm, z*cm, z*cm)
propagator = propagation_ASM
num_iters=501
linear_conv=False
medium=False
loss = nn.MSELoss().to(device)
bit=1
mode=1

target_amp_o=utils.loadtarget('./data/1.png',channel,dtype,device)
target_amp = utils.interpolate(target_amp_o, scale_factor=2, mode='nearest',linear_conv=linear_conv)
target_amp = utils.crop(target_amp, roi_res, medium=medium,linear_conv=linear_conv)
target_pha_o=torch.zeros_like(target_amp_o)
# target_pha=utils.QuaPhaseGen(slm_res,np.pi/slm_res[1],np.pi/slm_res[0],dtype,device)
target_pha = utils.interpolate(target_pha_o, scale_factor=2, mode='nearest',linear_conv=linear_conv)
target_pha = utils.crop(target_pha_o, roi_res, medium=medium,linear_conv=linear_conv)

real, imag = utils.polar_to_rect(target_amp_o,target_pha_o)
target= torch.complex(real, imag)

pyplot.figure('target_amp')
pyplot.imshow(target_amp.squeeze().cpu().detach().numpy(), cmap='gray')
pyplot.figure('target_pha')
pyplot.imshow(target_pha.squeeze().cpu().detach().numpy(), cmap='gray')
# pyplot.show()

sign = utils.LBSign.apply
relu = utils.LBRelu.apply

timeN=1
lr_amp=8e-2
lr_pha=2e-2

init_amp = (1.0 * torch.rand(timeN, 1, *slm_res)*2-1).to(device).requires_grad_(True)
# init_amp = utils.QuaPhaseGen(slm_res,np.pi/slm_res[1],np.pi/slm_res[0],dtype,device).requires_grad_(True)
# init_amp = torch.tensor(io.imread('./phases/SGD_ASM/red/ini/1_0.png') / 255,dtype=dtype).reshape(1, 1, *slm_res).to(device).requires_grad_(True)
# init_amp = (1.0 * torch.rand(timeN, 1, *slm_res)*2-1).to(device).requires_grad_(True)
# init_amp=(utils.IntAmp(slm_res,target_amp_o,'quadratic',0,dtype,device)).requires_grad_(True)
# init_pha = (1* np.pi * (torch.rand(timeN, 1, *slm_res)*2-1)).to(device).requires_grad_(True)
# init_pha=utils.QuaPhaseGen(slm_res,np.pi/slm_res[1],np.pi/slm_res[0],dtype,device).requires_grad_(True)

# optvars = [{'params': init_amp, 'lr': lr_amp},{'params': init_pha, 'lr': lr_pha}]
optvars = [{'params': init_amp, 'lr': lr_amp}]
# optvars = [{'params': init_pha, 'lr': lr_pha}]
optimizer = optim.Adam(optvars)

slm_N=torch.zeros_like(init_amp)
# slm_N=torch.zeros_like(init_pha)

uz=torch.zeros(1, 1, *target_pha.size()[-2:]).to(device)
L1loss = nn.L1Loss()

for k in range(num_iters):
    print(k)
    optimizer.zero_grad()
    recon_amp = torch.zeros_like(target_amp)
    for timen in range(timeN):
        binaryholo = utils.BitArraySGD(sign, relu, init_amp[timen:timen+1], bit, k, num_iters, mode, dtype, device)
        binaryholo_p = utils.pad(binaryholo, padval=0, stacked_complex=False, linear_conv=linear_conv)
        recon_field = utils.FourSpecGen(binaryholo_p, mode, bit)
        recon_field = utils.crop(recon_field, roi_res, medium=medium, linear_conv=linear_conv)
        recon_amp = recon_field.abs()
        # recon_amp += torch.pow(recon_field.abs(),2)
        recon_pha = utils.phaseCal(recon_field)

        if k == (num_iters - 1):
            slm_N[timen:timen + 1] = binaryholo

    # recon_amp = torch.sqrt(recon_amp/timeN)
    with torch.no_grad():
        s = (recon_amp * target_amp).mean() / (recon_amp ** 2).mean()  # scale minimizing MSE btw recon and

    mse_amp_loss = loss(s * recon_amp, target_amp)
    lossValue = mse_amp_loss

    loss_prior = L1loss(utils.laplacian(recon_pha), uz)
    # lossValue = lossValue + 0.01 * loss_prior

    lossValue.backward()
    optimizer.step()

psnr_loss = utils.psnr_func(target_amp, s * recon_amp, data_range=1)
ssim_loss = 1 - ssim_func(target_amp, s * recon_amp, data_range=2)
print('mse:', mse_amp_loss,'psnr:', psnr_loss, 'ssim:', 1 - ssim_loss,'loss_prior:', loss_prior)

utils.Saveholo(root_path, timeN, slm_N, 1, bit)
pyplot.figure('binaryholo')
pyplot.imshow(binaryholo.squeeze().cpu().detach().numpy(), cmap='gray')
recon_amp = (s*recon_amp).squeeze().cpu().detach().numpy()
recon_srgb = utils.srgb_lin2gamma(np.clip(recon_amp ** 2, 0.0, 1.0))
pyplot.figure('recon_srgb')
pyplot.imshow(recon_srgb, cmap='gray')
recon_pha = utils.phaseCal(recon_field)
pyplot.figure('recon_pha')
pyplot.imshow(recon_pha.squeeze().cpu().detach().numpy(), cmap='gray')
pyplot.show()