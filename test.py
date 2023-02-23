import torch
import matplotlib.pyplot as plt
dtype=torch.float32
# a = torch.arange(-100,100,1,dtype=dtype)
# # k=torch.arange(0,500,1,dtype=dtype)
# k = torch.tensor(0,dtype=dtype)
# num_iters = torch.tensor(500,dtype=dtype)
# s = 1-torch.tanh(k/ num_iters)
# b=torch.tanh(s*a)
# plt.plot(a,b)
# plt.show()

k = torch.tensor(450,dtype=dtype)
# k=torch.arange(0,500,1,dtype=dtype)
num_iters = torch.tensor(500,dtype=dtype)
s = 1-torch.tanh(k/ num_iters*10)
# plt.plot(k,s)
# plt.show()
print(s)