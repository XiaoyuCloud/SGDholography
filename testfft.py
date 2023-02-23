import torch
import torch.fft as tfft
import utils.utils as utils
import matplotlib.pyplot as pyplot

slm_res = (1600,2560)
device = torch.device('cuda')
dtype = torch.float32
channel =0
target_amp=utils.loadtarget('./data/1.png',channel,dtype,device)
# target_conj=utils.ConjImageGen(target_amp,'quadratic',0,dtype,device)

holo=utils.IntAmp(slm_res,target_amp,'quadratic',0,dtype,device)
holo=utils.ErrorDiffusion(holo,type='Floyd-Steinberg',dtype=dtype,numten='tensor4D',thresh=0.,min=-1.,max=1.)

pyplot.figure('holo')
pyplot.imshow(holo.squeeze().cpu().detach().numpy(), cmap='gray')
# pyplot.figure('holo直方图')
# pyplot.hist(holo.squeeze().cpu().detach().numpy())

recon=tfft.fftshift(tfft.fftn(holo, dim=(-2, -1)), (-2, -1))
recon[0,0,800-10:800+10,1280-10:1280+10]=0
recon_amp=recon.abs().squeeze().cpu().detach().numpy()
recon_pha=torch.angle(recon).squeeze().cpu().detach().numpy()


pyplot.figure('recon_amp')
pyplot.imshow(recon_amp, cmap='gray')
pyplot.figure('recon_pha')
pyplot.imshow(recon_pha, cmap='gray')
pyplot.show()
cc=1