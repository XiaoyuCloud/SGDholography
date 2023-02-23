"""
SGD holography
"""

import os
import torch
import torch.nn as nn
import configargparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import utils.utils as utils
from utils.modules import SGD
from propagation_ASM import propagation_ASM
import numpy as np
from skimage import io

# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--channel', type=int, default=0, help='Red:0, green:1, blue:2')
p.add_argument('--method', type=str, default='SGD', help='Type of algorithm, GS/SGD/DPAC/HOLONET/UNET')
p.add_argument('--prop_model', type=str, default='ASM', help='Type of propagation model, ASM or model')
p.add_argument('--root_path', type=str, default='./phases', help='Directory where optimized phases will be saved.')
p.add_argument('--data_path', type=str, default='./data', help='Directory for the dataset')
p.add_argument('--lr', type=float, default=8e-1, help='Learning rate for phase variables (for SGD)')#ori:8e-3,best:8e-2,会影响重建质量
p.add_argument('--lr_s', type=float, default=2e-1, help='Learning rate for learnable scale (for SGD)')#ori:2e-3,best:2e-2,会影响重建质量
p.add_argument('--num_iters', type=int, default=501, help='Number of iterations (GS, SGD)')#ori:500,best:1001,次：501
p.add_argument('--citl', type=utils.str2bool, default=False, help='Use of Camera-in-the-loop optimization with SGD')

# parse arguments
opt = p.parse_args()
run_id = f'{opt.method}_{opt.prop_model}'  # {algorithm}_{prop_model} format

channel = opt.channel  # Red:0 / Green:1 / Blue:2
chan_str = ('red', 'green', 'blue')[channel]

# Hyperparameters setting
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
z=20
prop_dist = (z * cm, z* cm, z* cm)[channel]  # propagation distance from SLM plane to target plane,ori:20,new:13.04
wavelength = (638 * nm, 520 * nm, 450 * nm)[channel]  # wavelength of each color,ori:638,DMD:632.992,GAEA2:637，520，450，fisba：错：660，488，405，对：638，520，450
feature_size = (7.56*1 * um, 7.56*1 * um)  # SLM pitch,ori:6.4,DMD:7.56,GAEA2:3.74
slm_res = (1600,2560)  # resolution of SLM,ori:1080,1920,DMD:1600,2560,GAEA2:2160,3840
image_res = (1600,2560)
roi_res = (1438,2300)  # regions of interest (to penalize for SGD)，boat:1294,2300,reso:1360,1582,flower:1438,2300，THU:280,640
dtype = torch.float32  # default datatype (Note: the result may be slightly different if you use float64, etc.)
device = torch.device('cuda')  # The gpu you are using
# target_dep=torch.tensor(io.imread('./data/depth/1d.png'),dtype=dtype).reshape(1, 1, *slm_res).to(device)
target_dep=None
timeN=1
bit=8

# Options for the algorithm
loss = nn.MSELoss().to(device)  # loss functions to use (try other loss functions!)
s0 = 1.0  # initial scale

root_path = os.path.join(opt.root_path, run_id, chan_str)  # path for saving out optimized phases

# Tensorboard writer
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
summaries_dir = os.path.join(root_path, 'summaries/')+ TIMESTAMP
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(summaries_dir)

camera_prop = None

# Simulation model
if opt.prop_model == 'ASM':
    propagator = propagation_ASM  # Ideal model

# Select Phase generation method, algorithm
if opt.method == 'SGD':
    phase_only_algorithm = SGD(prop_dist, wavelength, feature_size, opt.num_iters, roi_res, root_path,
                               opt.prop_model, propagator, loss, opt.lr, opt.lr_s, s0, opt.citl, camera_prop, writer, device)

init_phase = (1.0 * torch.rand(timeN, 1, *slm_res)*2-1).to(device)
# init_phase = (1.0 * torch.rand(timeN, 1, *slm_res)).to(device)
# init_phase = (1.0 * torch.rand(timeN, 1, *slm_res)*0).to(device)
# init_phase = (1.0 * torch.rand(timeN, 1, *slm_res)*2* np.pi).to(device)
# init_phase = ((1.0 * torch.rand(timeN, 1, *slm_res)-0.5)+ np.pi).to(device)
# init_phase = torch.tensor(io.imread('./phases/SGD_ASM/red/ini/1_0.png') / 255,dtype=dtype).reshape(1, 1, *slm_res).to(device)
target_amp=utils.loadtarget('./data/1.png',channel,dtype,device)
final_phase = phase_only_algorithm(target_amp, init_phase,target_dep,bit)

print(final_phase.shape)

utils.Saveholo(root_path, timeN, final_phase, 1, bit)
print(f'    - Done, result: --root_path={root_path}')
