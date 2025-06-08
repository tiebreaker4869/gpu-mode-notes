import torch

a = torch.randn((10000, 10000)).cuda()

square = torch.compile(torch.square)

square(a)