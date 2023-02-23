"""
utils
"""
import math
import numpy as np

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as pyplot
import GPUtil
from skimage import util,io
from imageio import imread
from pytorch_msssim import ssim as ssim_func
import scipy as sp
from scipy import special
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.fft as tfft

def mul_complex(t1, t2):
    """multiply two complex valued tensors element-wise. the two last dimensions are
    assumed to be the real and imaginary part

    complex multiplication: (a+bi)(c+di) = (ac-bd) + (bc+ad)i
    """
    # real and imaginary parts of first tensor
    a, b = t1.split(1, 4)
    # real and imaginary parts of second tensor
    c, d = t2.split(1, 4)

    # multiply out
    return torch.cat((a * c - b * d, b * c + a * d), 4)


def div_complex(t1, t2):
    """divide two complex valued tensors element-wise. the two last dimensions are
    assumed to be the real and imaginary part

    complex division: (a+bi) / (c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2) i
    """
    # real and imaginary parts of first tensor
    (a, b) = t1.split(1, 4)
    # real and imaginary parts of second tensor
    (c, d) = t2.split(1, 4)

    # get magnitude
    mag = torch.mul(c, c) + torch.mul(d, d)

    # multiply out
    return torch.cat(((a * c + b * d) / mag, (b * c - a * d) / mag), 4)


def reciprocal_complex(t):
    """element-wise inverse of complex-valued tensor

    reciprocal of complex number z=a+bi:
    1/z = a / (a^2 + b^2) - ( b / (a^2 + b^2) ) i
    """
    # real and imaginary parts of first tensor
    (a, b) = t.split(1, 4)

    # get magnitude
    mag = torch.mul(a, a) + torch.mul(b, b)

    # multiply out
    return torch.cat((a / mag, -(b / mag)), 4)


def rect_to_polar(real, imag):
    """Converts the rectangular complex representation to polar"""
    mag = torch.pow(real**2 + imag**2, 0.5)
    ang = torch.atan2(imag, real)
    return mag, ang


def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag

def polar_to_rect_mode(holobit,mode,bit):
    """Converts the polar complex representation to rectangular"""
    if mode==1:
        real, imag = polar_to_rect(holobit,torch.zeros_like(holobit))
    else:
        if bit==8:
            real, imag = polar_to_rect(torch.ones_like(holobit), holobit* 2 * np.pi)
        else:
            real, imag = polar_to_rect(torch.ones_like(holobit), holobit* 1/2 * np.pi)
    return real, imag

def replace_amplitude(field, amplitude):
    """takes a Complex tensor with real/imag channels, converts to
    amplitude/phase, replaces amplitude, then converts back to real/imag

    resolution of both Complex64 tensors should be (M, N, height, width)
    """
    # replace amplitude with target amplitude and convert back to real/imag
    real, imag = polar_to_rect(amplitude, field.angle())

    # concatenate
    return torch.complex(real, imag)

def replace_phase(field):
    """takes a Complex tensor with real/imag channels, converts to
    amplitude/phase, replaces amplitude, then converts back to real/imag

    resolution of both Complex64 tensors should be (M, N, height, width)
    """
    # replace amplitude with target amplitude and convert back to real/imag
    real, imag = polar_to_rect(field.abs(), field.angle())

    # concatenate
    return real

def ifftshift(tensor):
    """ifftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def fftshift(tensor):
    """fftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def ifft2(tensor_re, tensor_im, shift=False):
    """Applies a 2D ifft to the complex tensor represented by tensor_re and _im"""
    tensor_out = torch.stack((tensor_re, tensor_im), 4)

    if shift:
        tensor_out = ifftshift(tensor_out)
    (tensor_out_re, tensor_out_im) = torch.ifft(tensor_out, 2, True).split(1, 4)

    tensor_out_re = tensor_out_re.squeeze(4)
    tensor_out_im = tensor_out_im.squeeze(4)

    return tensor_out_re, tensor_out_im


def fft2(tensor_re, tensor_im, shift=False):
    """Applies a 2D fft to the complex tensor represented by tensor_re and _im"""
    # fft2
    (tensor_out_re, tensor_out_im) = torch.fft(torch.stack((tensor_re, tensor_im), 4), 2, True).split(1, 4)

    tensor_out_re = tensor_out_re.squeeze(4)
    tensor_out_im = tensor_out_im.squeeze(4)

    # apply fftshift
    if shift:
        tensor_out_re = fftshift(tensor_out_re)
        tensor_out_im = fftshift(tensor_out_im)

    return tensor_out_re, tensor_out_im


def roll_torch(tensor, shift, axis):
    """implements numpy roll() or Matlab circshift() functions for tensors"""
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


def pad_stacked_complex(field, pad_width, padval=0, mode='constant'):
    """Helper for pad_image() that pads a real padval in a complex-aware manner"""
    if padval == 0:
        pad_width = (0, 0, *pad_width)  # add 0 padding for stacked_complex dimension
        return nn.functional.pad(field, pad_width, mode=mode)
    else:
        if isinstance(padval, torch.Tensor):
            padval = padval.item()

        real, imag = field[..., 0], field[..., 1]
        real = nn.functional.pad(real, pad_width, mode=mode, value=padval)
        imag = nn.functional.pad(imag, pad_width, mode=mode, value=0)
        return torch.stack((real, imag), -1)


def pad_image(field, target_shape, pytorch=True, stacked_complex=True, padval=0, mode='constant'):
    size_diff = np.array(target_shape) - np.array(field.shape[-2:])
    odd_dim = np.array(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        pad_axes = [int(p)  # convert from np.int64
                    for tple in zip(pad_front[::-1], pad_end[::-1])
                    for p in tple]
        return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
    else:
        return field

def pad(u_in,padval,stacked_complex,linear_conv):
    if linear_conv:
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        u_in = pad_image(u_in, conv_size, padval=padval, stacked_complex=stacked_complex)
    return u_in

def interpolate(target_amp, scale_factor, mode,linear_conv):
    if not linear_conv:
        return target_amp
    else:
        target_amp = func.interpolate(target_amp, scale_factor=scale_factor, mode=mode)
        return target_amp

def crop_image(field, target_shape, pytorch=True, stacked_complex=True):
    if target_shape is None:
        return field

    size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
    odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        # crop_total = np.maximum(size_diff, 0)
        # crop_front = (crop_total + 1 - odd_dim) // 2
        # crop_end = (crop_total + odd_dim) // 2
        crop_front =np.array([81,130])
        crop_end=np.array([810,130])
        # crop_front = np.array([81*2, 130*2])
        # crop_end = np.array([810*2, 130*2])

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        return field[(..., *crop_slices)]
    else:
        return field

def crop(field, target_shape,medium,linear_conv):
    if target_shape is None:
        return field

    # crop dimensions that need to decrease in size
    if medium:
        target_shape = np.array(target_shape)
        if linear_conv:
            target_shape=target_shape*2
        size_diff = np.array(field.shape[-2:]) - target_shape
        odd_dim = np.array(field.shape[-2:]) % 2
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2
    else:
        if not linear_conv:
            crop_front =np.array([81,130])
            crop_end=np.array([810,130])
        else:
            crop_front = np.array([81*2, 130*2])
            crop_end = np.array([810*2, 130*2])

    crop_slices = [slice(int(f), int(-e) if e else None)
                   for f, e in zip(crop_front, crop_end)]
    return field[(..., *crop_slices)]

def srgb_gamma2lin(im_in):
    """converts from sRGB to linear color space"""
    thresh = 0.04045
    im_out = np.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055)**(2.4))
    return im_out


def srgb_lin2gamma(im_in):
    """converts from linear to sRGB color space"""
    thresh = 0.0031308
    im_out = np.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in**(1 / 2.4)) - 0.055)
    return im_out


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def phasemap_8bit(phasemap, inverted=True):
    """convert a phasemap tensor into a numpy 8bit phasemap that can be directly displayed

    Input
    -----
    :param phasemap: input phasemap tensor, which is supposed to be in the range of [-pi, pi].
    :param inverted: a boolean value that indicates whether the phasemap is inverted.

    Output
    ------
    :return: output phasemap, with uint8 dtype (in [0, 255])
    """

    # output_phase = ((phasemap + np.pi) % (2 * np.pi)) / (2 * np.pi)
    output_phase = (phasemap % (2 * np.pi)) / (2 * np.pi)
    if inverted:
        phase_out_8bit = ((1 - output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    else:
        phase_out_8bit = ((output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    return phase_out_8bit

def burst_img_processor(img_burst_list):
    img_tensor = np.stack(img_burst_list, axis=0)
    img_avg = np.mean(img_tensor, axis=0)
    return im2float(img_avg)  # changed from int8 to float32


def im2float(im, dtype=np.float32):
    """convert uint16 or uint8 image to float32, with range scaled to 0-1

    :param im: image
    :param dtype: default np.float32
    :return:
    """
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')


def propagate_field(input_field, propagator, prop_dist=0.2, wavelength=520e-9, feature_size=(6.4e-6, 6.4e-6),
                    prop_model='ASM', dtype=torch.float32, precomputed_H_exp=None,offset=0,target_dep=None):
    """
    A wrapper for various propagation methods, including the parameterized model.
    Note that input_field is supposed to be in Cartesian coordinate, not polar!

    Input
    -----
    :param input_field: pytorch complex tensor shape of (1, C, H, W), the field before propagation, in X, Y coordinates
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength of the wave in m.
    :param feature_size: pixel pitch
    :param prop_model: propagation model ('ASM', 'MODEL', 'fresnel', ...)
    :param trained_model: function or model instance for propagation
    :param dtype: torch.float32 by default
    :param precomputed_H: Propagation Kernel in Fourier domain (could be calculated at the very first time and reuse)

    Output
    -----
    :return: output_field: pytorch complex tensor shape of (1, C, H, W), the field after propagation, in X, Y coordinates
    """

    if prop_model == 'ASM':
        output_field = propagator(u_in=input_field, z=prop_dist, feature_size=feature_size, wavelength=wavelength,
                                  dtype=dtype, precomped_H_exp=precomputed_H_exp,offset=offset,target_dep=target_dep)

    return output_field


def write_sgd_summary(slm_phase, out_amp, target_amp, k,
                      writer=None, path=None, s=0., prefix='test'):
    """tensorboard summary for SGD

    :param slm_phase: Use it if you want to save intermediate phases during optimization.
    :param out_amp: PyTorch Tensor, Field amplitude at the image plane.
    :param target_amp: PyTorch Tensor, Ground Truth target Amplitude.
    :param k: iteration number.
    :param writer: SummaryWriter instance.
    :param path: path to save image files.
    :param s: scale for SGD algorithm.
    :param prefix:
    :return:
    """
    loss = nn.MSELoss().to(out_amp.device)
    loss_value = loss(s * out_amp, target_amp)
    psnr_value = psnr(target_amp.squeeze().cpu().detach().numpy(), (s * out_amp).squeeze().cpu().detach().numpy())
    ssim_value = ssim(target_amp.squeeze().cpu().detach().numpy(), (s * out_amp).squeeze().cpu().detach().numpy())

    s_min = (target_amp * out_amp).mean() / (out_amp**2).mean()
    psnr_value_min = psnr(target_amp.squeeze().cpu().detach().numpy(), (s_min * out_amp).squeeze().cpu().detach().numpy())
    ssim_value_min = ssim(target_amp.squeeze().cpu().detach().numpy(), (s_min * out_amp).squeeze().cpu().detach().numpy())

    if writer is not None:
        writer.add_image(f'{prefix}_Recon/amp', (s * out_amp).squeeze(0), k)
        writer.add_scalar(f'{prefix}_loss', loss_value, k)
        writer.add_scalar(f'{prefix}_psnr', psnr_value, k)
        writer.add_scalar(f'{prefix}_ssim', ssim_value, k)

        writer.add_scalar(f'{prefix}_psnr/scaled', psnr_value_min, k)
        writer.add_scalar(f'{prefix}_ssim/scaled', ssim_value_min, k)

        writer.add_scalar(f'{prefix}_scalar', s, k)


def write_gs_summary(slm_field, recon_field, target_amp, k, writer, roi=(880, 1600), prefix='test'):
    """tensorboard summary for GS"""
    slm_phase = slm_field.angle()
    recon_amp, recon_phase = recon_field.abs(), recon_field.angle()
    loss = nn.MSELoss().to(recon_amp.device)

    recon_amp = crop_image(recon_amp, target_shape=roi, stacked_complex=False)
    target_amp = crop_image(target_amp, target_shape=roi, stacked_complex=False)

    recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True)
                  / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))

    loss_value = loss(recon_amp, target_amp)
    psnr_value = psnr(target_amp.squeeze().cpu().detach().numpy(), recon_amp.squeeze().cpu().detach().numpy())
    ssim_value = ssim(target_amp.squeeze().cpu().detach().numpy(), recon_amp.squeeze().cpu().detach().numpy())

    if writer is not None:
        writer.add_image(f'{prefix}_Recon/amp', recon_amp.squeeze(0), k)
        writer.add_scalar(f'{prefix}_loss', loss_value, k)
        writer.add_scalar(f'{prefix}_psnr', psnr_value, k)
        writer.add_scalar(f'{prefix}_ssim', ssim_value, k)


def get_psnr_ssim(recon_amp, target_amp, multichannel=False,dtype='float32'):
    """get PSNR and SSIM metrics"""
    target_amp=np.array(target_amp, dtype=dtype)
    psnrs, ssims = {}, {}

    # amplitude
    psnrs['amp'] = psnr(target_amp, recon_amp)
    ssims['amp'] = ssim(target_amp, recon_amp, win_size=11, sigma=1.5,use_sample_covariance=False, gaussian_weights=True, multichannel=multichannel, data_range=2)

    # linear
    target_linear = target_amp**2
    recon_linear = recon_amp**2
    psnrs['lin'] = psnr(target_linear, recon_linear)
    ssims['lin'] = ssim(target_linear, recon_linear, win_size=11, sigma=1.5,use_sample_covariance=False, gaussian_weights=True, multichannel=multichannel, data_range=2)

    # srgb
    target_srgb = srgb_lin2gamma(np.clip(target_linear, 0.0, 1.0))
    recon_srgb = srgb_lin2gamma(np.clip(recon_linear, 0.0, 1.0))
    psnrs['srgb'] = psnr(target_srgb, recon_srgb)
    ssims['srgb'] = ssim(target_srgb, recon_srgb, win_size=11, sigma=1.5,use_sample_covariance=False, gaussian_weights=True, multichannel=multichannel, data_range=2)

    return psnrs, ssims

def get_psnr_ssim_time(recon_amp, target_amp, multichannel=False,dtype='float32',target_shape=None,psnrs_time=None,ssims_time=None):
    """get PSNR and SSIM metrics"""
    recon_amp=torch.sqrt(recon_amp)
    recon_amp = crop_image(recon_amp, target_shape=target_shape, stacked_complex=False)
    recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True)
                  / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))
    recon_amp = recon_amp.squeeze().cpu().detach().numpy()
    target_amp = target_amp.squeeze().cpu().detach().numpy()
    target_amp=np.array(target_amp, dtype=dtype)

    psnrs, ssims = {}, {}

    # amplitude
    psnrs['amp'] = psnr(target_amp, recon_amp)
    ssims['amp'] = ssim(target_amp, recon_amp, win_size=11, sigma=1.5,use_sample_covariance=False, gaussian_weights=True, multichannel=multichannel, data_range=2)

    # linear
    target_linear = target_amp**2
    recon_linear = recon_amp**2
    psnrs['lin'] = psnr(target_linear, recon_linear)
    ssims['lin'] = ssim(target_linear, recon_linear, win_size=11, sigma=1.5,use_sample_covariance=False, gaussian_weights=True, multichannel=multichannel, data_range=2)

    # srgb
    target_srgb = srgb_lin2gamma(np.clip(target_linear, 0.0, 1.0))
    recon_srgb = srgb_lin2gamma(np.clip(recon_linear, 0.0, 1.0))
    psnrs['srgb'] = psnr(target_srgb, recon_srgb)
    ssims['srgb'] = ssim(target_srgb, recon_srgb, win_size=11, sigma=1.5,use_sample_covariance=False, gaussian_weights=True, multichannel=multichannel, data_range=2)

    for domain in ['amp', 'lin', 'srgb']:
        psnrs_time[domain].append(psnrs[domain])
        ssims_time[domain].append(ssims[domain])

    return psnrs_time, ssims_time

def str2bool(v):
    """ Simple query parser for configArgParse (which doesn't support native bool from cmd)
    Ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def make_kernel_gaussian(sigma, kernel_size):

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2
    variance = sigma**2

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = ((1 / (2 * math.pi * variance))
                       * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)
                                   / (2 * variance)))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    return gaussian_kernel


def quantized_phase(phasemap):
    """
    just quantize phase into 8bit and return a tensor with the same dtype
    :param phasemap:
    :return:
    """

    # Shift to [0 1]
    phasemap = (phasemap + np.pi) / (2 * np.pi)

    # Convert into integer and take rounding
    phasemap = torch.round(255 * phasemap)

    # Shift to original range
    phasemap = phasemap / 255 * 2 * np.pi - np.pi
    return phasemap

def BitArrayGS(array,bit):
    if bit==1:
        # threshold=torch.mean(array)
        # array1bit =torch.ones_like(array)
        # array1bit[array<threshold] =0
        arrayNbit=torch.sign(array)
    elif bit==8:
        # arrayBit=(fieldNormal(array)*255).round()/255
        arrayBit = fieldNormal(array)
        # arrayBit = array
    else:
        #array1bit = fieldNormal(array)  #当bit=16，即优化目标全息图完全灰度，而非8bit量化时，归一化会造成横向噪声而不归一化则不会
        arrayBit =array
    return arrayBit

def BitArraySGD(sign,relu,slm_phase, bit,k,num_iters,mode,dtype,device):
    if k >= 0 and bit == 1:
        if mode==1:
            slm_phaseBit = (sign(slm_phase,k,num_iters,dtype,device) + 1) / 2
        else:
            slm_phaseBit = (slm_phase % (2 * np.pi) - np.pi) / (1 * np.pi)
            slm_phaseBit = sign(slm_phaseBit, k, num_iters, dtype, device)
    elif k >= 0 and bit == 8:
        if mode==1:
            slm_phaseBit = slm_phase
        else:
            slm_phaseBit = (slm_phase % (2 * np.pi)) / (2 * np.pi)
    return slm_phaseBit

def quantization(input,level,dtype,device):
    if level==1:
        thresh1 = torch.tensor(0, dtype=dtype).to(device)
        thresh2 = torch.tensor(1, dtype=dtype).to(device)
        input = torch.where(input <= 0.5, thresh1, thresh2)
    else:
        for i in range(level):
            torch.tensor(i/level, dtype=dtype).to(device)
            thresh1 = torch.tensor(i/level, dtype=dtype).to(device)
            thresh2 = torch.tensor((i+1)/level, dtype=dtype).to(device)
            input = torch.where((input >= thresh1) & (input < thresh2), thresh1,input)
    return input

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input,k,num_iters,dtype,device):
        ret=torch.sign(input)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        ret = grad_output.clamp_(-1, 1)
        return ret,None,None,None,None

class LBSignSim(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ret=torch.sign(input)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        ret = grad_output.clamp_(-1, 1)
        return ret

def signK(input,sign,k,num_iters,dtype,device):   #实测效果最好
    input = torch.tanh(input)
    delta = k / (num_iters-1)
    output = torch.where((input > -delta) & (input < delta), sign(input,k,num_iters,dtype,device) * delta, input)
    output = (output + 1) / 2
    return output

def signKrelu(input,sign,relu,k,num_iters): #实测效果不好
    input = fieldNormal(relu(input))
    delta = k / ((num_iters-1)*2)
    output = torch.where((input > (0.5-delta)) & (input < (0.5+delta)), sign(input-0.5)*delta+0.5, input)
    return output

class LBRelu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)

def reluT(input,sign,relu):
    input=relu(input)
    output = torch.where(input > 1, sign(input), input)
    return output

def fieldDynshow(field,delaytime):
    field_abs = field.abs()
    field_abs = field_abs.squeeze().cpu().detach().numpy()
    pyplot.clf()
    pyplot.imshow(field_abs, cmap='gray')
    pyplot.pause(delaytime)
    pyplot.ioff()

def fieldshow(field):
    field_abs = field.abs()
    field_abs = field_abs.squeeze().cpu().detach().numpy()
    pyplot.figure()
    pyplot.imshow(field_abs, cmap='gray')
    pyplot.show()

def phaseDynshow(field,delaytime):
    field_angle = field.angle()
    field_angle = field_angle.squeeze().cpu().detach().numpy()
    pyplot.clf()
    pyplot.imshow(field_angle, cmap='gray')
    pyplot.pause(delaytime)
    pyplot.ioff()

def phaseshow(field):
    field_angle = field.angle()
    field_angle = field_angle.squeeze().cpu().detach().numpy()
    # field_angle = np.unwrap(field_angle)
    pyplot.figure()
    pyplot.imshow(field_angle, cmap='gray')
    pyplot.show()

def valueDynshow(value,delaytime):
    value = value.squeeze().cpu().detach().numpy()
    pyplot.clf()
    pyplot.imshow(value, cmap='gray')
    pyplot.pause(delaytime)
    pyplot.ioff()

def valueshow(value):
    value = value.squeeze().cpu().detach().numpy()
    pyplot.figure()
    pyplot.imshow(value, cmap='gray')
    pyplot.show()

def ReconfieldDynshow(field,target_amp,delaytime):
    recon_amp = field.abs()
    recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True)
                  / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))
    recon_amp = recon_amp.squeeze().cpu().detach().numpy()
    recon_srgb = srgb_lin2gamma(np.clip(recon_amp**2, 0.0, 1.0))
    pyplot.clf()
    pyplot.imshow(recon_srgb, cmap='gray')
    pyplot.pause(delaytime)
    pyplot.ioff()

def Reconfieldshow(recon_amp,target_amp):
    recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True)
                  / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))
    ssim_val_me = ssim_func(target_amp, recon_amp, data_range=2)
    psnr_val_me = psnr_func(target_amp, recon_amp, data_range=1)
    print('psnr_me:', psnr_val_me, 'ssim_me:', ssim_val_me)
    recon_amp = recon_amp.squeeze().cpu().detach().numpy()
    recon_srgb = srgb_lin2gamma(np.clip(recon_amp ** 2, 0.0, 1.0))
    pyplot.figure()
    pyplot.imshow(recon_srgb, cmap='gray')
    pyplot.show()

def RefWaveGen(slm_field,RefWaveTheta,wavelength,prop_dist,dtype):
    X,Y,N, M = slm_field.shape
    theta = RefWaveTheta / 180 * math.pi
    RefWave = torch.complex(torch.ones((X,Y,N,M),dtype=dtype),torch.tensor(0,dtype=dtype))
    k=2*math.pi/wavelength/1000
    for n in range(N):
        RefWave[:,:,n,:]=torch.exp(1j * k * (n+1) * torch.sin(torch.tensor(theta,dtype=dtype)))

    return RefWave

def FieldAddRefWave(slm_field,RefWaveTheta,wavelength,prop_dist,dtype):
    RefWave = RefWaveGen(slm_field,RefWaveTheta,wavelength,prop_dist,dtype)
    if prop_dist < 0:
        slm_field = slm_field * torch.conj(RefWave).to(slm_field.device)
    else:
        slm_field= slm_field * RefWave.to(slm_field.device)
    return slm_field

def SSfilterGen(u_in,prop_dist,dtype,linear_conv,offset):
    X, Y, N, M = u_in.shape
    SSfilter=torch.zeros((X,Y,N,M),dtype=dtype)
    bandn=350
    bandm=600
    if linear_conv:
        bandn=bandn*2
        bandm = bandm * 2
        offset=offset*2

    zoN=np.int(np.ceil(N / 2))
    zoM=np.int(np.ceil(M / 2))
    if offset==0:
        # SSfilter[:, :, zoN-bandn:zoN+bandn,zoM-bandm:zoM+bandm] = 1#离轴
        SSfilter=torch.tensor(1.,dtype=dtype)#离轴
    else:
        if abs(offset)<9999:
            # SSfilter[:, :, zoN + offset:zoN + bandn, zoM - bandm:zoM + bandm] = 1  # 同轴
            SSfilter[:, :, zoN + offset:zoN - offset, zoM + offset:zoM - offset] = 1  # 同轴
        # SSfilter[:, :, zoN + offset:, :] = 1  # 同轴
        else:
            SSfilter=torch.tensor(1.,dtype=dtype)
    return SSfilter

def CirclefilterGen(field_resolution,dtype,linear_conv,r,device):
    if r==0:
        Circlefilter = torch.ones(1,1,*field_resolution).to(device)
    else:
        num_y, num_x = field_resolution[0], field_resolution[1]
        x = np.linspace(-num_x / 2, num_x / 2 - 1, num_x)
        y = np.linspace(-num_y / 2, num_y / 2 - 1, num_y)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X ** 2 + Y ** 2)
        if linear_conv:
            r=r*2
        R = torch.tensor(R, dtype=dtype).to(device)
        R = torch.reshape(R, (1, 1, *R.size()))
        Circlefilter = torch.zeros(1,1,*field_resolution).to(device)
        Circlefilter[R<r]=1
    return Circlefilter

def BandLimitedHolo(array):
    arrayF = ifftshift(torch.fft.fftn(ifftshift(array), dim=(-2, -1)))
    x,y,n,m=array.shape
    Filter = torch.zeros(x,y,n, m)
    bandn=350*1
    bandm = 600*1
    zoN=1202   #-0.0091,[1202,1281]350；
    zoM=1281
    # zoN = np.int(np.ceil(n / 2))
    # zoM=np.int(np.ceil(m / 2))
    Filter[:,:,zoN-bandn:zoN+bandn,zoM-bandm:zoM+bandm]=1          #离轴
    #Filter[:, :, zoN-10:zoN + band, zoM-band:zoM + band] = 1            #同轴
    #Filter[:, :, zoN+10:zoN + bandn, zoM - bandm:zoM + bandm] = 1  # 同轴
    # Filter[:, :, zoN-20:zoN + bandn, zoM - bandm:zoM + bandm] = 1  # 同轴，考虑0级，滤掉大部分共轭像，效果较好。
    #Filter[:, :, zoN - 20:n, :] = 1  # 同轴,不加带宽约束，效果小茶
    arrayF=arrayF*Filter.to(arrayF.device)
    array = fftshift(torch.fft.ifftn(fftshift(arrayF), dim=(-2, -1)))
    return array


def functional_conv2d(im,dtype,device):
    sobel_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],dtype=dtype).to(device)  #
    weight = sobel_kernel.reshape((1, 1, 3, 3)).requires_grad_(True)
    edge_detect = func.conv2d(im, weight)
    return edge_detect

def phaseCal(field):
    real=torch.real(field)
    imag=torch.imag(field)
    mag,angle=rect_to_polar(real, imag)
    return angle

# def unwrap(angle):
#     angleTensor=angle
#     angle=unwrap_phase(angle.squeeze(0))
#     angle=torch.tensor(np.unwrap(angle.cpu().numpy()),angleTensor.dtype).to(angleTensor.device)
#     return angle

def unwrap(x):
    y = x % (2 * np.pi)
    return torch.where(y > np.pi, 2*np.pi - y, y)

def fieldNormal(field):
    norm_field=(field-field.min())/(field.max()-field.min())
    return norm_field

def fieldNormalNozero(field):
    norm_field=field/field.max()
    return norm_field

def bitTofield(field,bit):
    if bit==8:
        field=field/255
    return field

def captured_amp_revise(captured_amp,recon_amp):
    captured_amp=(recon_amp.max()-recon_amp.min())*captured_amp+recon_amp.min()
    return captured_amp

def get_gpu_info(self, text = ""):
    print("当前行为为：", text)
    GPUtil.showUtilization()

def one_unit8(holo):
    # holo=(holo+1)/2
    holo_255=(holo*255).round()
    holo_unit8 = holo_255.squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    return holo_unit8

def Saveholo(root_path,timeN,final_phase,target_idx,bit):
    cond_mkdir(root_path)
    for timen in range(timeN):
        final_phase_timen=final_phase[timen, 0, :, :]
        if bit==1:
            # holo_bit = util.img_as_bool(((final_phase[timen, 0, :, :]+1)/2).cpu().detach())
            holo_bit = util.img_as_bool(final_phase_timen.cpu().detach())
            cv2.imwrite(os.path.join(root_path, f'{target_idx}_{timen}.png'), holo_bit.astype(int),[cv2.IMWRITE_PNG_BILEVEL, bit])
        elif bit==8:
            holo_bit = one_unit8(final_phase_timen.cpu().detach())
            cv2.imwrite(os.path.join(root_path, f'{target_idx}_{timen}.png'), holo_bit)

def psnr_func(image_true, image_test, data_range=1):
    err = func.mse_loss(image_true, image_test)
    return 10 * torch.log10((data_range ** 2) / err)

# def gradient(input):
#     n,m=
def TVnorm(input):
    xg=torch.roll(input, shifts=(0, 0,-1,0), dims=(0,1,2,3))-input
    yg = torch.roll(input, shifts=(0, 0,0,-1), dims=(0,1,2,3))- input
    # TVnorm=torch.sum(xg.abs())+torch.sum(yg.abs())
    TVnorm = torch.sum(torch.sqrt(xg.abs()**2 + yg.abs()**2))
    TVnorm=TVnorm/(input.size(2)*input.size(3))
    return TVnorm


class Round(torch.autograd.Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class Int(torch.autograd.Function):

    @staticmethod
    def forward(self, input):
        output = input.int()
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift =  cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des

def get_good_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def siftImageAlignment(img1, img2):
    _, kp1, des1 = sift_kp(img1)
    _, kp2, des2 = sift_kp(img2)
    goodMatch = get_good_match(des1, des2)
    if len(goodMatch) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold);
        # 其中H为求得的单应性矩阵矩阵
        # status则返回一个列表来表征匹配成功的特征点。
        # ptsA,ptsB为关键点
        # cv2.RANSAC, ransacReprojThreshold这两个参数与RANSAC有关
        imgOut = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return imgOut, H, status

def max(list01):
    max_value=list01[0]
    for i in range(1, len(list01)):
        if max_value < list01[i]:
            max_value = list01[i]
    return max_value

def gamma_inverse(im):
    low_val = im <= 0.04045
    im[low_val] = 25 / 323 * im[low_val]
    im[np.logical_not(low_val)] = ((200 * im[np.logical_not(low_val)] + 11)
                                   / 211) ** (12 / 5)
    im = np.sqrt(im)  # to amplitude
    return im

def loadtarget(path,channel,dtype,device):
    target_amp = io.imread(path)
    if channel<3:
        target_amp = target_amp[..., channel, np.newaxis]
    else:
        target_amp = target_amp[..., :3]
    target_amp = im2float(target_amp, dtype=np.float64)
    target_amp=srgb_gamma2lin(target_amp)
    target_amp = np.sqrt(target_amp)
    target_amp = np.transpose(target_amp, axes=(2, 0, 1))
    target_amp = torch.tensor(target_amp, dtype=dtype).unsqueeze(0).to(device)
    return target_amp

def resultshow(slm_phaseBit,mode,bit,RefWaveTheta, wavelength, prop_dist, dtype,propagator,feature_size,prop_model,precomputed_H_exp,offset,roi_res,s,target_amp):
    real, imag = polar_to_rect_mode(slm_phaseBit,mode,bit)
    slm_field = torch.complex(real, imag)
    slm_field = FieldAddRefWave(slm_field, RefWaveTheta, wavelength, prop_dist, dtype)
    recon_field = propagate_field(slm_field, propagator, prop_dist, wavelength, feature_size, prop_model, dtype,
                                        precomputed_H_exp, offset)
    recon_amp = recon_field.abs()
    recon_amp = crop_image(recon_amp, target_shape=roi_res, stacked_complex=False)
    recon_amp_s = recon_amp * s
    Reconfieldshow(recon_amp_s, target_amp)
    phaseshow(recon_field)

def laplacian(img):

    # signed angular difference
    grad_x1, grad_y1 = grad(img, next_pixel=True)  # x_{n+1} - x_{n}
    grad_x0, grad_y0 = grad(img, next_pixel=False)  # x_{n} - x_{n-1}

    laplacian_x = grad_x1 - grad_x0  # (x_{n+1} - x_{n}) - (x_{n} - x_{n-1})
    laplacian_y = grad_y1 - grad_y0

    return laplacian_x + laplacian_y

def grad(img, next_pixel=False, sovel=False):
    if img.shape[1] > 1:
        permuted = True
        img = img.permute(1, 0, 2, 3)
    else:
        permuted = False

    # set diff kernel
    if sovel:  # use sovel filter for gradient calculation
        k_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32) / 8
        k_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32) / 8
    else:
        if next_pixel:  # x_{n+1} - x_n
            k_x = torch.tensor([[0, -1, 1]], dtype=torch.float32)
            k_y = torch.tensor([[1], [-1], [0]], dtype=torch.float32)
        else:  # x_{n} - x_{n-1}
            k_x = torch.tensor([[-1, 1, 0]], dtype=torch.float32)
            k_y = torch.tensor([[0], [1], [-1]], dtype=torch.float32)

    # upload to gpu
    k_x = k_x.to(img.device).unsqueeze(0).unsqueeze(0)
    k_y = k_y.to(img.device).unsqueeze(0).unsqueeze(0)

    # boundary handling (replicate elements at boundary)
    img_x = func.pad(img, (1, 1, 0, 0), 'replicate')
    img_y = func.pad(img, (0, 0, 1, 1), 'replicate')

    # take sign angular difference
    grad_x = signed_ang(func.conv2d(img_x, k_x))
    grad_y = signed_ang(func.conv2d(img_y, k_y))

    if permuted:
        grad_x = grad_x.permute(1, 0, 2, 3)
        grad_y = grad_y.permute(1, 0, 2, 3)

    return grad_x, grad_y

def signed_ang(angle):
    """
    cast all angles into [-pi, pi]
    """
    return (angle + math.pi) % (2*math.pi) - math.pi

# Adapted from https://github.com/svaiter/pyprox/blob/master/pyprox/operators.py
def soft_thresholding(x, gamma):
    """
    return element-wise shrinkage function with threshold kappa
    """
    return torch.maximum(torch.zeros_like(x),
                         1 - gamma / torch.maximum(torch.abs(x), 1e-10*torch.ones_like(x))) * x

def mod(x,y):
    return x- torch.floor(x/y)*y

class LBWhere(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input,dtype,device):
        thresh1 = torch.tensor(0, dtype=dtype).to(device)
        thresh2 = torch.tensor(1, dtype=dtype).to(device)
        ret=torch.where(input <= 0, thresh1, thresh2)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        ret = grad_output.clamp_(-1, 1)
        return ret,None,None

def BinaryHoloLee(target,ox,oy,kx,ky,device,dtype,where,sign):
    target_amp = abs(target)
    targer_pha = phaseCal(target)
    pha_cri = abs(mod(targer_pha / 2 / math.pi + kx * ox + ky * oy, 1) - 0.5)
    amp_cri = torch.arcsin(target_amp) / 2 / math.pi
    BinaryHolo =where(amp_cri-pha_cri,dtype,device)
    # BinaryHolo =ErrorDiffusion(fieldNormal(amp_cri-pha_cri),type='Floyd-Steinberg',dtype=dtype,numten='tensor4D')

    # pha_cri = torch.cos(2 * math.pi * (kx * ox + ky * oy) - targer_pha)
    # amp_cri = torch.cos(math.pi * target_amp)
    # BinaryHolo = where(amp_cri,pha_cri, dtype, device)

    # BinaryHolo = sign(amp_cri-pha_cri)
    # BinaryHolo=(BinaryHolo+1)/2
    return BinaryHolo,amp_cri,pha_cri

def LGmode(slm_res, feature_size,p,l, w,device,dtype):
    X, Y=XYGen(slm_res=slm_res, feature_size=feature_size, numten='numpy', device=device, dtype=dtype)
    r = np.sqrt(X ** 2 + Y ** 2)
    phi = np.arctan2(Y, X)
    Phasesor = np.exp(-1j * phi * l)
    Amplitude = ((r / w) * np.sqrt(2)) ** (abs(l)) * np.exp(-(r / w) ** 2) *sp.special.eval_genlaguerre(p,abs(l),2*(r/w)**2)
    Norma = np.sqrt(2 / (np.pi * (math.factorial(l))))
    Field = Norma * Phasesor * Amplitude
    return Field

def XYGen(slm_res, feature_size,numten,device,dtype):
    if numten== 'numpy':
        x = np.linspace(-slm_res[1] / 2, slm_res[1] / 2 - 1, slm_res[1]) * feature_size[1]
        y = np.linspace(-slm_res[0] / 2, slm_res[0] / 2 - 1, slm_res[0]) * feature_size[0]
        X, Y = np.meshgrid(x, y)
    elif numten== 'tensor':
        x = torch.linspace(-slm_res[1] / 2, slm_res[1] / 2 - 1, slm_res[1],dtype=dtype,device=device) * feature_size[1]
        y = torch.linspace(-slm_res[0] / 2, slm_res[0] / 2 - 1, slm_res[0],dtype=dtype,device=device) * feature_size[0]
        Y, X = torch.meshgrid(y, x)
    return X,Y

def Hprecom(field_resolution, wavelength,feature_size,dtype,z,Hexpz,device):
    num_y, num_x = field_resolution[0], field_resolution[1]
    dy, dx = feature_size
    y, x = (dy * float(num_y), dx * float(num_x))
    fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / y, num_y)
    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / x, num_x)
    FX, FY = np.meshgrid(fx, fy)
    H_exp = 2 * math.pi * np.sqrt(1 / wavelength ** 2 - (FX ** 2 + FY ** 2))
    H_exp = torch.tensor(H_exp, dtype=dtype).to(device)
    H_exp = torch.reshape(H_exp, (1, 1, *H_exp.size()))
    if Hexpz=='True':
        H_exp = torch.mul(H_exp, z)
        H = torch.exp(1j * H_exp)
        return H
    else:
        return H_exp

def ErrorDiffusionBlock(err,errblock,thresh,max,min,fc):
    xc = float(err)
    xm = float(err)
    xc = xc > thresh
    if min == 0:
        t = int(xc) * max
    else:
        t = (int(xc) * 2 - 1) * max
    e = xm - t
    fc1 = fc * e
    errblock = errblock + fc1
    return t,errblock

def ErrorDiffusion(im1,type,dtype,numten,thresh,min,max):
    if numten=='numpy':
        im2 = np.zeros((im1.shape[0], im1.shape[1]), dtype=dtype)
        if type=='Floyd-Steinberg':
            err = np.zeros((im1.shape[0] + 2, im1.shape[1] + 2), dtype=dtype)
            err[1:im1.shape[0] + 1, 1:im1.shape[1] + 1] = im1
            fc = np.array([[0, 0, 7.0], [3.0, 5.0, 1.0]], dtype=np.float32) / [16]
            for i in range(1, im1.shape[0]):
                for j in range(1, im1.shape[1]):
                    if err[i, j] < thresh:
                        im2[i - 1, j - 1] = min
                        e = err[i, j]-min
                    else:
                        im2[i - 1, j - 1] = max
                        e = err[i, j] - max
                    im2[i-1, j-1],err[i:i + 2, j-1:j + 2]=ErrorDiffusionBlock(err[i, j], err[i:i + 2, j-1:j + 2], thresh, max, min, fc)
        elif type=='Jarvis-Jidice-Ninke':
            err = np.zeros((im1.shape[0] + 4, im1.shape[1] + 4), dtype=dtype)
            err[2:im1.shape[0] + 2, 2:im1.shape[1] + 2] = im1
            fc = np.array([[0,0,0,7.0,5.0], [3.0,5.0,7.0,5.0,3.0],[1.0,3.0,5.0,3.0,1.0]], dtype=np.float32) / [48]
            for i in range(2, im1.shape[0] + 2):
                for j in range(2, im1.shape[1] + 2):
                    # if err[i, j] < thresh:
                    #     im2[i - 2, j - 2] = min
                    #     e = err[i, j]-min
                    # else:
                    #     im2[i - 2, j - 2] = max
                    #     e = err[i, j] - max
                    # err[i, j + 1] = err[i, j + 1] + 7 * e / 48
                    # err[i, j + 2] = err[i, j + 2] + 5 * e / 48
                    # err[i + 1, j - 2] = err[i + 1, j - 2] + 3 * e / 48
                    # err[i + 1, j - 1] = err[i + 1, j - 1] + 5 * e / 48
                    # err[i + 1, j] = err[i + 1, j] + 7 * e / 48
                    # err[i + 1, j + 1] = err[i + 1, j + 1] + 5 * e / 48
                    # err[i + 1, j + 2] = err[i + 1, j + 2] + 3 * e / 48
                    # err[i + 2, j - 2] = err[i + 2, j - 2] + 1 * e / 48
                    # err[i + 2, j - 1] = err[i + 2, j - 1] + 3 * e / 48
                    # err[i + 2, j] = err[i + 2, j] + 5 * e / 48
                    # err[i + 2, j + 1] = err[i + 2, j + 1] + 3 * e / 48
                    # err[i + 2, j + 2] = err[i + 2, j + 2] + 1 * e / 48

                    # xc = float(err[i, j])
                    # xm = float(err[i, j])
                    # xc = xc > thresh
                    # if min == 0:
                    #     t = int(xc) * max
                    # else:
                    #     t = (int(xc) * 2 - 1) * max
                    # im2[i, j] = t
                    # e = np.subtract(xm, t)
                    # fc1 = np.multiply(fc, e)
                    # err[i:i + 3, j - 2:j + 3] = err[i:i + 3, j - 2:j + 3] + fc1
                    im2[i - 2, j - 2], err[i:i + 3, j - 2:j + 3] = ErrorDiffusionBlock(err[i, j],err[i:i + 3, j - 2:j + 3],thresh, max, min, fc)
    if numten=='tensor2D':
        im2 = torch.zeros((im1.shape[2], im1.shape[3]), dtype=dtype).to(im1.device)
        if type=='Floyd-Steinberg':
            err = torch.zeros((im1.shape[2] + 2, im1.shape[3] + 2), dtype=dtype).to(im1.device)
            err[1:im1.shape[2] + 1, 1:im1.shape[3] + 1] = im1
            fc = torch.tensor([[0, 0, 7.0], [3.0, 5.0, 1.0]], dtype=dtype,device=im1.device) / 16
            for i in range(1, im1.shape[2]):
                for j in range(1, im1.shape[3]):
                    if err[i, j] < thresh:
                        im2[i - 1, j - 1] = min
                        e = err[i, j]
                    else:
                        im2[i - 1, j - 1] = max
                        e = err[i, j] - max
                    # err[i, j + 1] = err[i, j + 1] + 7 * e / 16
                    # err[i + 1, j - 1] = err[i + 1, j - 1] + 3 * e / 16
                    # err[i + 1, j] = err[i + 1, j] + 5 * e / 16
                    # err[i + 1, j + 1] = err[i + 1, j + 1] + e / 16
                    im2[i - 1, j - 1], err[i:i + 2, j - 1:j + 2] = ErrorDiffusionBlock(err[i, j],err[i:i + 2, j - 1:j + 2],thresh, max, min, fc)
        elif type=='Jarvis-Jidice-Ninke':
            err = torch.zeros((im1.shape[2] + 4, im1.shape[3] + 4), dtype=dtype).to(im1.device)
            err[2:im1.shape[2] + 2, 2:im1.shape[3] + 2] = im1
            fc = torch.tensor([[0,0,0,7.0,5.0], [3.0,5.0,7.0,5.0,3.0],[1.0,3.0,5.0,3.0,1.0]], dtype=dtype, device=im1.device) / 48
            for i in range(2, im1.shape[2] + 2):
                for j in range(2, im1.shape[3] + 2):
                    if err[i, j] < thresh:
                        im2[i - 2, j - 2] = min
                        e = err[i, j]
                    else:
                        im2[i - 2, j - 2] = max
                        e = err[i, j] - max
                    err[i, j + 1] = err[i, j + 1] + 7 * e / 48
                    err[i, j + 2] = err[i, j + 2] + 5 * e / 48
                    err[i + 1, j - 2] = err[i + 1, j - 2] + 3 * e / 48
                    err[i + 1, j - 1] = err[i + 1, j - 1] + 5 * e / 48
                    err[i + 1, j] = err[i + 1, j] + 7 * e / 48
                    err[i + 1, j + 1] = err[i + 1, j + 1] + 5 * e / 48
                    err[i + 1, j + 2] = err[i + 1, j + 2] + 3 * e / 48
                    err[i + 2, j - 2] = err[i + 2, j - 2] + 1 * e / 48
                    err[i + 2, j - 1] = err[i + 2, j - 1] + 3 * e / 48
                    err[i + 2, j] = err[i + 2, j] + 5 * e / 48
                    err[i + 2, j + 1] = err[i + 2, j + 1] + 3 * e / 48
                    err[i + 2, j + 2] = err[i + 2, j + 2] + 1 * e / 48
        im2=im2.unsqueeze(0).unsqueeze(0)
    elif numten== 'tensor4D':
        im2 = torch.zeros_like(im1)
        if type == 'Floyd-Steinberg':
            err = torch.zeros(1,1,im1.shape[2] + 2, im1.shape[3] + 2, dtype=dtype).to(im1.device)
            err[0,0,1:im1.shape[2] + 1, 1:im1.shape[3] + 1] = im1
            fc = torch.tensor([[0, 0, 7.0], [3.0, 5.0, 1.0]], dtype=dtype, device=im1.device) / 16
            for i in range(1, im1.shape[2]):
                for j in range(1, im1.shape[3]):
                    # if err[0,0,i, j] < thresh:
                    #     im2[0,0,i - 1, j - 1] = min
                    #     e = err[0,0,i, j]-min
                    # else:
                    #     im2[0,0,i - 1, j - 1] = max
                    #     e = err[0,0,i, j] - max
                    # err_effect_im2
                    # err[0,0,i, j + 1] = err[0,0,i, j + 1] + 7 * e / 16
                    # err[0,0,i + 1, j - 1] = err[0,0,i + 1, j - 1] + 3 * e / 16
                    # err[0,0,i + 1, j] = err[0,0,i + 1, j] + 5 * e / 16
                    # err[0,0,i + 1, j + 1] = err[0,0,i + 1, j + 1] + e / 16
                    im2[0,0,i - 1, j - 1], err[0,0,i:i + 2, j - 1:j + 2] = ErrorDiffusionBlock(err[0,0,i, j],err[0,0,i:i + 2, j - 1:j + 2],thresh, max, min, fc)
        elif type == 'Jarvis-Jidice-Ninke':
            err = torch.zeros(1,1,im1.shape[2] + 4, im1.shape[3] + 4, dtype=dtype).to(im1.device)
            err[0,0,2:im1.shape[2] + 2, 2:im1.shape[3] + 2] = im1
            fc = torch.tensor([[0, 0, 0, 7.0, 5.0], [3.0, 5.0, 7.0, 5.0, 3.0], [1.0, 3.0, 5.0, 3.0, 1.0]], dtype=dtype,device=im1.device) / 48
            for i in range(2, im1.shape[2] + 2):
                for j in range(2, im1.shape[3] + 2):
                    if err[0,0,i, j] < thresh:
                        im2[0,0,i - 2, j - 2] = min
                        e = err[0,0,i, j]
                    else:
                        im2[0,0,i - 2, j - 2] = max
                        e = err[0,0,i, j] - max
                    err[0,0,i, j + 1] = err[0,0,i, j + 1] + 7 * e / 48
                    err[0,0,i, j + 2] = err[0,0,i, j + 2] + 5 * e / 48
                    err[0,0,i + 1, j - 2] = err[0,0,i + 1, j - 2] + 3 * e / 48
                    err[0,0,i + 1, j - 1] = err[0,0,i + 1, j - 1] + 5 * e / 48
                    err[0,0,i + 1, j] = err[0,0,i + 1, j] + 7 * e / 48
                    err[0,0,i + 1, j + 1] = err[0,0,i + 1, j + 1] + 5 * e / 48
                    err[0,0,i + 1, j + 2] = err[0,0,i + 1, j + 2] + 3 * e / 48
                    err[0,0,i + 2, j - 2] = err[0,0,i + 2, j - 2] + 1 * e / 48
                    err[0,0,i + 2, j - 1] = err[0,0,i + 2, j - 1] + 3 * e / 48
                    err[0,0,i + 2, j] = err[0,0,i + 2, j] + 5 * e / 48
                    err[0,0,i + 2, j + 1] = err[0,0,i + 2, j + 1] + 3 * e / 48
                    err[0,0,i + 2, j + 2] = err[0,0,i + 2, j + 2] + 1 * e / 48
    return im2

def QuaPhaseGen(slm_res,a,b,dtype,device):
    m = torch.linspace(-slm_res[1] / 2, slm_res[1] / 2 - 1, slm_res[1], dtype=dtype, device=device)
    n = torch.linspace(-slm_res[0] / 2, slm_res[0] / 2 - 1, slm_res[0], dtype=dtype, device=device)
    n, m = torch.meshgrid(n, m)
    quapha=a*m**2+b*n**2
    quapha = torch.reshape(quapha, (1, 1, *slm_res)) % (2 * np.pi)
    return quapha

def FourSpecGen(input,mode, bit):
    # if input.dtype==torch.float:
    #     real, imag = polar_to_rect_mode(input, mode, bit)
    #     input = torch.complex(real, imag)
    # input_f = ifftshift(torch.fft.fftn(ifftshift(input), dim=(-2, -1)))
    input_f = tfft.fftshift(tfft.fftn(input, dim=(-2, -1)), (-2, -1))
    return input_f

def ZerosCom(N,M,dtype,device):
    real=torch.zeros(N, M, dtype=dtype, device=device)
    imag=torch.zeros(N, M, dtype=dtype, device=device)
    zeroscom = torch.complex(real, imag)
    return zeroscom

def ConjImageGen(target_amp,target_pha,medpint,dtype,device):
    [Nori, Mori] = target_amp.size()[-2:]
    conjimage = torch.zeros(Nori, Mori, dtype=torch.complex64, device=device)
    N_lrc = int((Nori / 2))
    M_lrc = int((Mori - 1))
    if target_pha=='quadratic':
        m = torch.linspace(int((-M_lrc + 1) / 2), int((M_lrc + 1) / 2 - 1), M_lrc, dtype=dtype, device=device)
        n = torch.linspace(N_lrc - 1, 0, N_lrc, dtype=dtype, device=device)
        n, m = torch.meshgrid(n, m)
        target_pha = np.pi / M_lrc * m** 2 + np.pi / N_lrc * n**2
    elif target_pha=='rand':
        target_pha = 2 * np.pi * torch.rand(N_lrc, M_lrc, dtype=dtype, device=device)
    else:
        target_pha = torch.zeros(N_lrc, M_lrc, dtype=dtype, device=device)
    conjimage_lrc_up = target_amp[0,0,0:N_lrc, 0: M_lrc] * torch.exp(1j * target_pha)
    conjimage_lrc_down = torch.conj(torch.flip(conjimage_lrc_up, [0,1]))
    conjimage_lrc = torch.zeros(2 * N_lrc - 1, M_lrc, dtype=torch.complex64, device=device)
    conjimage_lrc[0: N_lrc-1,:]=conjimage_lrc_up[0: N_lrc-1,:]
    conjimage_lrc[N_lrc-1, 0: int((M_lrc - 1) / 2)]=conjimage_lrc_up[N_lrc-1, 0: int((M_lrc - 1) / 2)]
    conjimage_lrc[N_lrc-1, int((M_lrc + 1) / 2-1)] = medpint
    conjimage_lrc[N_lrc-1, int((M_lrc + 1) / 2) : M_lrc]=conjimage_lrc_down[0, int((M_lrc + 1) / 2) : M_lrc]
    conjimage_lrc[N_lrc : 2 * N_lrc - 1,:]=conjimage_lrc_down[1: N_lrc,:]
    conjimage[1: Nori, 1: Mori]=conjimage_lrc
    conjimage=conjimage.unsqueeze(0).unsqueeze(0)
    return conjimage

def IntAmp(slm_res,target_amp,target_pha,medpint,dtype,device):
    target_conj = ConjImageGen(target_amp, target_pha, medpint, dtype, device)
    init_amp = tfft.ifftn(tfft.ifftshift(target_conj, (-2, -1)), dim=(-2, -1)) * (slm_res[0] * slm_res[1])
    init_amp = torch.real(init_amp)
    init_amp = fieldNormalNozero(init_amp)
    return init_amp