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
medium=True
loss = nn.MSELoss().to(device)

target_amp_o=utils.loadtarget('./data/1.png',channel,dtype,device)
target_amp = utils.interpolate(target_amp_o, scale_factor=2, mode='nearest',linear_conv=linear_conv)
target_amp = utils.crop(target_amp, roi_res, medium=medium,linear_conv=linear_conv)
target_pha_o=torch.zeros_like(target_amp_o)
# target_pha=utils.QuaPhaseGen(slm_res,np.pi/slm_res[1],np.pi/slm_res[0],dtype,device)
target_pha = utils.interpolate(target_pha_o, scale_factor=2, mode='nearest',linear_conv=linear_conv)
target_pha = utils.crop(target_pha_o, roi_res, medium=medium,linear_conv=linear_conv)
# target_pha=utils.loadtarget('./data/1.jpg',channel,dtype,device)*2*np.pi-np.pi
# target_amp=torch.ones_like(target_pha)
# p=1
# l = 1
# w = 1.5e-3
# LGbeam=utils.LGmode(slm_res, feature_size,p,l, w,device,dtype)
# target_amp=torch.tensor(abs(LGbeam),dtype=dtype,device=device).unsqueeze(0).unsqueeze(0)
# target_pha=torch.tensor(np.angle(LGbeam),dtype=dtype,device=device).unsqueeze(0).unsqueeze(0)
real, imag = utils.polar_to_rect(target_amp_o,target_pha_o)
target= torch.complex(real, imag)
pyplot.figure('target_amp')
pyplot.imshow(target_amp.squeeze().cpu().detach().numpy(), cmap='gray')
pyplot.figure('target_pha')
pyplot.imshow(target_pha.squeeze().cpu().detach().numpy(), cmap='gray')
# pyplot.show()

# ky=0
offset=0
# offset=torch.tensor(-100.0,requires_grad=True, device=device,dtype=dtype)
# filter= utils.SSfilterGen(target, -prop_dists[channel], dtype, linear_conv, offset).to(device)
filter=utils.CirclefilterGen(slm_res,dtype,linear_conv,abs(offset), device)

where = utils.LBWhere.apply
sign = utils.LBSign.apply
relu = utils.LBRelu.apply

pyplot.figure('filter')
pyplot.imshow(filter.abs().squeeze().cpu().detach().numpy(), cmap='gray')
# pyplot.show()

m = torch.linspace(1, slm_res[1], slm_res[1], dtype=dtype, device=device)
n = torch.linspace(1, slm_res[0], slm_res[0], dtype=dtype, device=device)
n, m = torch.meshgrid(n, m)
ox,oy=utils.XYGen(slm_res, feature_size, 'tensor', device, dtype)
H=utils.Hprecom(slm_res, wavelengths[channel],feature_size,dtype,prop_dists[channel],'True',device)

# lr_period=2e-1
period=torch.tensor(12.0,requires_grad=False, device=device,dtype=dtype)
# optvars = [{'params': period, 'lr': lr_period}]
kx = 1.0 / (period * feature_size[1])
ky = 1.0 / (period * feature_size[0])
# kx=0
# ky=0

# kx=slm_res[1]/4*1/(slm_res[1]*feature_size[1])
# ky=slm_res[0]/4*1/(slm_res[0]*feature_size[0])

timeN=1
lr_amp=8e-2
lr_pha=2e-2
# init_amp = (1.0 * torch.rand(timeN, 1, *slm_res)*2-1).to(device).requires_grad_(True)
# init_amp = utils.QuaPhaseGen(slm_res,np.pi/slm_res[1],np.pi/slm_res[0],dtype,device).requires_grad_(True)
# init_amp = torch.tensor(io.imread('./phases/SGD_ASM/red/ini/1_0.png') / 255,dtype=dtype).reshape(1, 1, *slm_res).to(device).requires_grad_(True)
init_pha = (1* np.pi * (torch.rand(timeN, 1, *slm_res)*2-1)).to(device).requires_grad_(True)
# init_pha=utils.QuaPhaseGen(slm_res,np.pi/slm_res[1],np.pi/slm_res[0],dtype,device).requires_grad_(True)
# optvars = [{'params': init_amp, 'lr': lr_amp},{'params': init_pha, 'lr': lr_pha}]
# optvars = [{'params': init_amp, 'lr': lr_amp}]
optvars = [{'params': init_pha, 'lr': lr_pha}]

bit=1
mode=0
s = torch.tensor(1., requires_grad=True, device=device)
optimizer = optim.Adam(optvars)
# slm_N=torch.zeros_like(init_amp)
slm_N=torch.zeros_like(init_pha)

u = torch.zeros(1, 1, *target_pha.size()[-2:]).to(device)
z = torch.zeros(1, 1, *target_pha.size()[-2:]).to(device)

for k in range(num_iters):
    print(k)
    optimizer.zero_grad()
    recon_amp = torch.zeros_like(target_amp)
    for timen in range(timeN):
        binaryholo = utils.BitArraySGD(sign, relu, init_pha[timen:timen+1], bit, k, num_iters, mode, dtype, device)
        binaryholo_p = utils.pad(binaryholo, padval=0, stacked_complex=False, linear_conv=linear_conv)
        real, imag = utils.polar_to_rect_mode(binaryholo_p,mode,bit)

        # init_real, init_imag = utils.polar_to_rect(init_amp, init_pha)
        # ini_com = torch.complex(init_real, init_imag)
        # binaryholo, amp_cri, pha_cri = utils.BinaryHoloLee(ini_com, ox, oy, kx, ky, device, dtype, where,sign)
        # real, imag = utils.polar_to_rect(binaryholo,torch.zeros_like(binaryholo))

        slm_field= torch.complex(real, imag)

        # if k==(num_iters-1):
        #     U1_bef = utils.ifftshift(torch.fft.fftn(utils.ifftshift(slm_field), dim=(-2, -1)))
        #     pyplot.figure('U1_bef')
        #     pyplot.imshow(utils.fieldNormal(U1_bef.abs()).squeeze().cpu().detach().numpy(), cmap='gray',vmin = 0, vmax = 0.001)

        # ref_real, ref_imag = utils.polar_to_rect(1, -2 * np.pi * (kx * m * feature_size[1]+ky * n * feature_size[0]))
        # RefWave= torch.complex(ref_real, ref_imag).unsqueeze(0).unsqueeze(0)
        # slm_field= slm_field * RefWave

        # slm_field=ini_com

        U1 = utils.ifftshift(torch.fft.fftn(utils.ifftshift(slm_field), dim=(-2, -1)))
        # recon_field = utils.fftshift(torch.fft.ifftn(utils.fftshift(U1 * filter * H), dim=(-2, -1)))
        recon_field =U1
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

    L1loss = nn.L1Loss()
    loss_prior = L1loss(utils.laplacian(recon_pha), (z - u))
    # lossValue = lossValue + 0.01 * loss_prior

    lossValue.backward()
    optimizer.step()

psnr_loss = utils.psnr_func(target_amp, s * recon_amp, data_range=1)
ssim_loss = 1 - ssim_func(target_amp, s * recon_amp, data_range=2)
print('mse:', mse_amp_loss,'psnr:', psnr_loss, 'ssim:', 1 - ssim_loss,'loss_prior:', loss_prior)

utils.Saveholo(root_path, timeN, slm_N, 1, bit)
pyplot.figure('binaryholo')
pyplot.imshow(binaryholo.squeeze().cpu().detach().numpy(), cmap='gray')
# pyplot.figure('f4')
# pyplot.plot(binaryholo[:,:,int(slm_res[0]/2), int(slm_res[1]/4):int(slm_res[1]/2)].squeeze().cpu().detach().numpy())
# pyplot.plot(amp_cri[:,:,int(slm_res[0]/2), int(slm_res[1]/4):int(slm_res[1]/2)].squeeze().cpu().detach().numpy())
# pyplot.plot(pha_cri[:,:,int(slm_res[0]/2), int(slm_res[1]/4):int(slm_res[1]/2)].squeeze().cpu().detach().numpy())
recon_amp = (s*recon_amp).squeeze().cpu().detach().numpy()
recon_srgb = utils.srgb_lin2gamma(np.clip(recon_amp ** 2, 0.0, 1.0))
pyplot.figure('recon_srgb')
pyplot.imshow(recon_srgb, cmap='gray')
recon_pha = utils.phaseCal(recon_field)
pyplot.figure('recon_pha')
pyplot.imshow(recon_pha.squeeze().cpu().detach().numpy(), cmap='gray')
pyplot.show()
cc=1