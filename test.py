import torch
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x**2
print(y)
y.backward(torch.ones_like(x))
# y.backward(torch.ones(2, 2))




