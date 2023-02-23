import torch
class Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = torch.exp(i)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result

exp = Exp()
x1 = torch.tensor([3.], requires_grad=True)
x2 = exp.apply(x1)
x3=x2*2
x4=x3*4

x4.backward()
print(x1.grad)
