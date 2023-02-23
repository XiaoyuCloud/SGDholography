import torch
a = torch.range(1, 4).reshape(2, 2)
b = torch.stack([a,a,a], dim=0)
c = b - 2
d = torch.where((a>3) | (a<4), b, c)
print('a:', a)
print('b:', b)
print('c:', c)
print('d:', d)
