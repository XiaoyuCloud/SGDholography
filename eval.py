"""
eval
"""

import imageio
import os
from skimage import io
import scipy.io as sio
import torch
import numpy as np
import configargparse
from propagation_ASM import propagation_ASM
import utils.utils as utils
from pytorch_msssim import ssim as ssim_func
import torch.nn.functional as func
import matplotlib.pyplot as pyplot
import json

# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--channel', type=int, default=0, help='red:0, green:1, blue:2, rgb:3')
p.add_argument('--prop_model', type=str, default='ASM',
               help='Type of propagation model for reconstruction: ASM / MODEL / CAMERA')
p.add_argument('--root_path', type=str, default='./phases/SGD_ASM', help='Directory where test phases are being stored.')

# Parse
opt = p.parse_args()
channel = opt.channel
chs = range(channel) if channel == 3 else [channel]  # retrieve all channels if channel is 3
run_id = f'{opt.root_path.split("/")[-1]}_{opt.prop_model}'  # {algorithm}_{prop_model}

# Hyperparameters setting
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
chan_strs = ('red', 'green', 'blue', 'rgb')
z0=20
target_length=0.2
n=20
I=0
z=z0+I/n*target_length
prop_dists = (z*cm, z*cm, z*cm) #ori:20,new:12.04
wavelengths = (638 * nm, 520 * nm, 450 * nm)  # wavelength of each color,ori:638,DMD:632.992,GAEA2:637，fisba：660，488，405，对：638，520，450
feature_size = (7.56*um,7.56*um)  # SLM pitch,ori:6.4,DMD:7.56,GAEA2:3.74

# Resolutions
slm_res = (1600,2560)  # resolution of SLM,ori:1080,1920,DMD:1600,2560,GAEA2:2160,3840
if 'HOLONET' in run_id.upper():
    slm_res = (1072, 1920)
elif 'UNET' in run_id.upper():
    slm_res = (1024, 2048)

image_res = (1600,2560)
roi_res = (1438,2300)  # regions of interest (to penalize)，ori:880,1600,new:1438,2300
dtype = torch.float32  # default datatype (Note: the result may be slightly different if you use float64, etc.)
npdtype='float32'
device = torch.device('cuda')  # The gpu you are using

# You can pre-compute kernels for fast-computation
precomputed_H = [None] * 3
if opt.prop_model == 'ASM':
    propagator = propagation_ASM

print(f'  - reconstruction with {opt.prop_model}... ')

# Data path
data_path = './data'
recon_path = './recon'

# Placeholders for metrics
psnrs = {'amp': [], 'lin': [], 'srgb': []}
ssims = {'amp': [], 'lin': [], 'srgb': []}
idxs = []

psnrs_time = {'amp': [], 'lin': [], 'srgb': []}
ssims_time = {'amp': [], 'lin': [], 'srgb': []}

target_idx=1
recon_amp = []

timeN=24
offset = -10000
RefWaveTheta = 0
mode=1
bit=8
recon_amp_c=0
target_amp = utils.loadtarget('./data/1.png', channel, dtype, device)
# target_amp = func.interpolate(target_amp, scale_factor=2, mode='nearest')
target_amp = utils.crop_image(target_amp, target_shape=roi_res, stacked_complex=False).to(device)
# for each channel, propagate wave from the SLM plane to the image plane and get the reconstructed image.
for c in chs:
    # load and invert phase (our SLM setup)
    for timen in range(timeN):
        phase_filename = os.path.join(opt.root_path, chan_strs[c], f'{target_idx}_{timen}.png')
        slm_phase = io.imread(phase_filename) / 255
        slm_phase = torch.tensor(slm_phase, dtype=dtype).reshape(1, 1, *slm_res).to(device)
        slm_phase = utils.quantization(slm_phase, 1,dtype, device)
        # pyplot.figure('slm_phase')
        # pyplot.imshow(slm_phase.squeeze().cpu().detach().numpy(), cmap='gray')
        # pyplot.show()

        # propagate field
        real, imag = utils.polar_to_rect_mode(slm_phase, mode,bit)
        # real, imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase * 1 * np.pi)
        slm_field = torch.complex(real, imag)
        # slm_field  = utils.pad(slm_field , padval=0, stacked_complex=False, linear_conv=True)
        # slm_field = utils.FieldAddRefWave(slm_field, RefWaveTheta, wavelengths[c],-prop_dists[c],dtype)

        # recon_field = utils.propagate_field(slm_field, propagator, prop_dists[c], wavelengths[c], feature_size,
        #                                     opt.prop_model, dtype,offset=offset)
        recon_field =utils.ifftshift(torch.fft.fftn(utils.ifftshift(slm_field), dim=(-2, -1)))

        # cartesian to polar coordinate
        recon_amp_c += torch.pow(recon_field.abs(),2)
        psnrs_time, ssims_time = utils.get_psnr_ssim_time(recon_amp_c, target_amp, multichannel=(channel == 3), dtype=npdtype,target_shape=roi_res,psnrs_time=psnrs_time,ssims_time=ssims_time)
    jsObj_psnr = json.dumps(psnrs_time)
    fileObject = open('./recon/psnrs_time_dir.json', 'w')
    fileObject.write(jsObj_psnr)
    fileObject.close()
    jsObj_ssim = json.dumps(ssims_time)
    fileObject = open('./recon/ssims_time_dir.json', 'w')
    fileObject.write(jsObj_ssim)
    fileObject.close()
    pyplot.figure(1),pyplot.plot(psnrs_time['amp']),pyplot.show()
    pyplot.figure(2),pyplot.plot(ssims_time['amp']),pyplot.show()
    recon_amp_c=torch.sqrt(recon_amp_c/timeN)
    # recon_amp_c = recon_field.abs()

    # crop to ROI
    recon_amp_crop = utils.crop_image(recon_amp_c, target_shape=roi_res, stacked_complex=False)

    # append to list
    recon_amp.append(recon_amp_crop)

# list to tensor, scaling
recon_amp = torch.cat(recon_amp, dim=1)
recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True)
              / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))
ssim_val_me = ssim_func(target_amp,recon_amp,data_range=2)
psnr_val_me = utils.psnr_func(target_amp, recon_amp, data_range=1)
print('psnr_me:', psnr_val_me, 'ssim_me:', ssim_val_me)
# tensor to numpy
recon_amp_np = recon_amp.squeeze().cpu().detach().numpy()
target_amp_np = target_amp.squeeze().cpu().detach().numpy()

if channel == 3:
    recon_amp_np = recon_amp_np.transpose(1, 2, 0)
    target_amp_np = target_amp_np.transpose(1, 2, 0)

# calculate metrics
psnr_val, ssim_val = utils.get_psnr_ssim(recon_amp_np, target_amp_np, multichannel=(channel == 3), dtype=npdtype)

idxs.append(target_idx)

for domain in ['amp', 'lin', 'srgb']:
    psnrs[domain].append(psnr_val[domain])
    ssims[domain].append(ssim_val[domain])
    print(f'PSNR({domain}): {psnr_val[domain]},  SSIM({domain}): {ssim_val[domain]:.4f}, ')

# save reconstructed image in srgb domain
# recon_srgb = recon_amp_np
recon_srgb = utils.srgb_lin2gamma(np.clip(recon_amp_np**2, 0.0, 1.0))
utils.cond_mkdir(recon_path)
imageio.imwrite(os.path.join(recon_path, f'{target_idx}_{run_id}_{chan_strs[channel]}.png'), (recon_srgb * np.iinfo(np.uint8).max).round().astype(np.uint8))

# save it as a .mat file
data_dict = {}
data_dict['img_idx'] = idxs
for domain in ['amp', 'lin', 'srgb']:
    data_dict[f'ssims_{domain}'] = ssims[domain]
    data_dict[f'psnrs_{domain}'] = psnrs[domain]

sio.savemat(os.path.join(recon_path, f'metrics_{run_id}_{chan_strs[channel]}.mat'), data_dict)
