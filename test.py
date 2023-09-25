import torch
from cuda_rwkv import cuda_wkv
import torch
import numpy as np
from numba import cuda
import time
B = 2
T = 3
C = 4096
w = torch.randn(C,device='cuda', dtype=torch.float32)
u =torch.randn(C,device='cuda',dtype=torch.float32)
k = torch.randn((B,C),device='cuda',dtype=torch.float16)
v = torch.randn((B,C),device='cuda',dtype=torch.float16)
y = torch.empty(T*C,device='cuda',dtype=torch.float16)


aa = torch.randn((B, C),device='cuda',dtype=torch.float32)
bb = torch.randn((B, C),device='cuda',dtype=torch.float32)
pp = torch.randn((B, C),device='cuda',dtype=torch.float32)
lens = [1,2]
lens = torch.tensor(lens)
numset = torch.cumsum(lens, dim=0)
lens_gpu = lens.to(device=w.device,dtype=torch.int)
numset_gpu = numset.to(device=w.device,dtype=torch.int)
# device = torch.device('cuda:0')  # 选择所需的GPU设备
# lens_gpu = torch.tensor(lens, dtype=int, device=device)
# numset_gpu = torch.tensor(numset, dtype=int, device=device)
w1 = w.clone()
u1 = u.clone()
k1 = k.clone()
v1 = v.clone()
y1 = y.clone()
aa1 = aa.clone()
bb1 = bb.clone()
pp1 = pp.clone()
s = time.time()
cuda_wkv(B, C, w, u, k, v, y, aa, bb, pp, lens_gpu, numset_gpu)
y = y.reshape(T, C)
torch.cuda.synchronize()
e = time.time()
print(f'第一次cuda： {e-s}')


s = time.time()
cuda_wkv(B, C, w, u, k, v, y, aa, bb, pp, lens_gpu, numset_gpu)
y = y.reshape(T, C)
torch.cuda.synchronize()
e = time.time()
print(f'第二次cuda： {e-s}')



