import torch
import matplotlib.pyplot as plt

def quantization(input,dtype,device):
    level=2**4
    for i in range(level):
        torch.tensor(i/level, dtype=dtype).to(device)
        thresh1 = torch.tensor(i/level, dtype=dtype).to(device)
        thresh2 = torch.tensor((i+1)/level, dtype=dtype).to(device)
        input = torch.where((input >= thresh1) & (input < thresh2), thresh1,input)
    return input

dtype=torch.float32
device=torch.device('cpu')
input=torch.arange(0,1,0.01,dtype=dtype)
output=quantization(input,dtype,device)
plt.plot(input,output)
plt.show()
