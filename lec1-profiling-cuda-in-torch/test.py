import torch

t = torch.randn((1000, 1000)).cuda()

for _ in range(10):
    _ = torch.square(t)