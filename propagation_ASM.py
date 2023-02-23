"""
angular spectrum method (ASM)
"""

import math
import torch
import numpy as np
import utils.utils as utils
import torch.fft
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    def __init__(self, LayerN):
        self.len = LayerN
        self.data =  torch.arange(0, LayerN, 1)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.len

def propagation_ASM_3D_kernel(target_dep,z,n,LayerN,H_exp_ori,U1,ssfilter,STEP,target_length,conv_size,padval):
    device_i = n.device
    target_dep_I = target_dep.to(device_i)
    target_outOutline = torch.zeros([1, 1, 1600, 2560]).to(device_i)
    target_outOutline[target_dep_I == 0] = 1./LayerN.to(device_i)
    #target_outOutline[target_dep_I == 0] = 1.
    H_exp_ori_I = H_exp_ori.to(device_i)
    U1_I = U1.to(device_i)
    ssfilter_I = ssfilter.to(device_i)
    u_out_I=0
    for m in n:
        Maska = target_dep_I >= m * STEP
        Maskb = target_dep_I < (m + 1) * STEP
        zI = -(abs(z) + m  / LayerN * target_length)
        Mask = (Maska & Maskb).long() + target_outOutline
        Mask = utils.pad_image(Mask, conv_size, padval=padval, stacked_complex=False)
        H = torch.exp(1j * torch.mul(H_exp_ori_I, zI))
        # LayerIphaseOffset = torch.exp(-1j * k * (LayerZ - z));
        u_out_I += utils.fftshift(torch.fft.ifftn(utils.fftshift(H * U1_I * ssfilter_I), dim=(-2, -1))) * Mask
        del Maska, Maskb, zI, Mask, H
        torch.cuda.empty_cache()
    return u_out_I

class Model(nn.Module):
    def __init__(self,target_dep,z,LayerN,H_exp_ori,U1,ssfilter,STEP,target_length,conv_size,padval):
        super(Model, self).__init__()
        self.target_dep = target_dep
        self.z=z
        self.LayerN=LayerN
        self.H_exp_ori=H_exp_ori
        self.U1=U1
        self.ssfilter=ssfilter
        self.STEP=STEP
        self.target_length=target_length
        self.conv_size=conv_size
        self.padval=padval

    def forward(self, input):
        output = propagation_ASM_3D_kernel(self.target_dep,self.z,input,self.LayerN,self.H_exp_ori,self.U1,self.ssfilter,self.STEP,self.target_length,self.conv_size,self.padval)
        return output

def propagation_ASM(u_in, feature_size, wavelength, z, linear_conv=False,
                    padtype='zero', return_H=False, precomped_H=None,
                    return_H_exp=False, precomped_H_exp=None,
                    dtype=torch.float32,offset=0,target_dep=None):
    """Propagates the input field using the angular spectrum method

    Inputs
    ------
    u_in: PyTorch Complex tensor (torch.cfloat) of size (num_images, 1, height, width) -- updated with PyTorch 1.7.0
    feature_size: (height, width) of individual holographic features in m
    wavelength: wavelength in m
    z: propagation distance
    linear_conv: if True, pad the input to obtain a linear convolution
    padtype: 'zero' to pad with zeros, 'median' to pad with median of u_in's
        amplitude
    return_H[_exp]: used for precomputing H or H_exp, ends the computation early
        and returns the desired variable
    precomped_H[_exp]: the precomputed value for H or H_exp
    dtype: torch dtype for computation at different precision

    Output
    ------
    tensor of size (num_images, 1, height, width, 2)
    """
    #u_in = utils.pad_image(u_in, conv_size, padval=padval, stacked_complex=False)
    if linear_conv:
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in**2).sum(-1), 0.5))
        u_in = utils.pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

    if precomped_H is None and precomped_H_exp is None:
        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        field_resolution = u_in.size()

        # number of pixels
        num_y, num_x = field_resolution[2], field_resolution[3]

        # sampling inteval size
        dy, dx = feature_size

        # size of the field
        y, x = (dy * float(num_y), dx * float(num_x))

        # frequency coordinates sampling,存疑，稍后研究
        #fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
        #fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)
        fy = np.linspace(-1 / (2 * dy) , 1 / (2 * dy) -  1/ y, num_y)
        fx = np.linspace(-1 / (2 * dx) , 1 / (2 * dx) -  1/ x, num_x)
        # momentum/reciprocal space
        FX, FY = np.meshgrid(fx, fy)

        # transfer function in numpy (omit distance)
        HH = 2 * math.pi * np.sqrt(1 / wavelength**2 - (FX**2 + FY**2))

        # create tensor & upload to device (GPU)
        H_exp = torch.tensor(HH, dtype=dtype).to(u_in.device)

        ###
        # here one may iterate over multiple distances, once H_exp is uploaded on GPU

        # reshape tensor and multiply
        H_exp_ori = torch.reshape(H_exp, (1, 1, *H_exp.size()))

    # handle loading the precomputed H_exp value, or saving it for later runs
    elif precomped_H_exp is not None:
        H_exp_ori = precomped_H_exp

    U1 = utils.ifftshift(torch.fft.fftn(utils.ifftshift(u_in), dim=(-2, -1)))
    ssfilter = utils.SSfilterGen(U1, z, dtype, linear_conv, offset).to(U1.device)
    ssfilter.requires_grad = False
    if target_dep is not None:
        target_length = 2e-3
        batch_size = 5
        n = torch.tensor(batch_size)
        deviceN=torch.tensor(4)
        LayerN = n*deviceN
        STEP = 256 / LayerN
        dataset = RandomDataset(LayerN)
        rand_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        model = Model(target_dep,z,LayerN,H_exp_ori,U1,ssfilter,STEP,target_length,conv_size,padval)
        model = nn.DataParallel(model)
        model.to(U1.device)
        u_out = 0
        for data in rand_loader:
            input = data.to(U1.device)
            u_out_I = model(input)
            u_out += torch.sum(u_out_I, dim=0)
            # utils.get_gpu_info("内存情况")
        u_out=u_out.unsqueeze(0)
    else:
        if precomped_H is None:
            # multiply by distance
            H_exp = torch.mul(H_exp_ori, z)
            H = torch.exp(1j * H_exp)
        else:
            H = precomped_H

        if return_H_exp:
            return H_exp_ori
        if return_H:
            return H

        # U2 = H * U1
        # ssfilter = utils.SSfilterGen(U1, z, dtype, linear_conv, offset).to(U1.device)
        # U2 = U2 * ssfilter
        # u_out = utils.fftshift(torch.fft.ifftn(utils.fftshift(U2), dim=(-2, -1)))
        u_out =U1
        # u_out = utils.fftshift(torch.fft.ifftn(utils.fftshift(U1 * ssfilter), dim=(-2, -1)))
    # if linear_conv:
    #     return utils.crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)  # using complex tensor
    # else:
        return u_out