from skimage import io
import torch
import matplotlib.pyplot as pyplot
import cv2
import scipy.io as io
import wmi

w = wmi.WMI()
monitors = w.Win32_DesktopMonitor()
for m in monitors:
	print(m)

slm_res = (1600,2560)
device = torch.device('cuda')
dtype = torch.float32

# target_amp= torch.tensor(io.imread('./data/1.png')[:,:,0] / 255).reshape(1, 1, *slm_res).to(device)

m = torch.linspace(0, 255, slm_res[1], dtype=dtype, device=device)
n = torch.linspace(0, 255, slm_res[0], dtype=dtype, device=device)
n, m = torch.meshgrid(n, m)
target_amp=torch.reshape(m, (1, 1, *slm_res))
io.savemat('./recon/target_amp.mat', {'target_amp': target_amp.squeeze().squeeze().cpu().detach().numpy()})

pyplot.figure('target_amp')
pyplot.imshow(target_amp.squeeze().cpu().detach().numpy(),cmap='gray',vmin=0,vmax=1)
pyplot.show()

# cv2.imshow('target_amp',target_amp.squeeze().cpu().detach().numpy())
# key = cv2.waitKey(0)

