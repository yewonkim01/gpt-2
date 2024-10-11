import torch
import numpy as np
from torch.nn import functional as F

a = torch.randn(8,2,4,2)
x = F.softmax(a, dim=-1)
print(x.shape)
print(x.sum(dim=-1).shape)



