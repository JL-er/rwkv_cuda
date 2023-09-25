from numba import cuda, float32, int32
import numpy as np
import math
import torch
@cuda.jit
def kernel_wkv(C, _w, _u, _k, _v, _y, _aa, _bb, _pp, _lens, numset):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    _b = idx // C
    _c = idx % C
    _b_offset = 0
    if _b-1 >= 0:
        #_b_offset = (_lens[_b-1])*C
        _b_offset = (numset[_b-1])*C
    _offset = _b_offset
    _state_offset = _b * C + _c
    tim = _lens[_b]

    u = _u[_c]
    w = _w[_c]
    k = _k  #[B, C]
    v = _v
    y = _y
    aa = _aa[_state_offset]
    bb = _bb[_state_offset]
    pp = _pp[_state_offset]
    for i in range(tim):
        ii = i * C + _offset + _c
        kk = float32(k[ii])
        vv = float32(v[ii])
        ww = u + kk
        p = max(pp, ww)
        e1 = math.exp(pp - p)
        e2 = math.exp(ww - p)
        y[ii] = (e1 * aa + e2 * vv) / (e1 * bb + e2)
        ww = w + pp
        p = max(ww, kk)
        e1 = math.exp(ww - p)
        e2 = math.exp(kk - p)
        aa = e1 * aa + e2 * vv
        bb = e1 * bb + e2
        pp = p

    _aa[_state_offset] = aa
    _bb[_state_offset] = bb
    _pp[_state_offset] = pp

@cuda.jit
def kernel_mm_seq_fp16i8(B, N, M, x, w, mx, rx, my, ry, y, xs, ws, ys):
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    k = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
    if i < B and k < M:
        y_local = 0.0
        for j in range(N):
            y_local += np.float16(x[i * xs + j]) * (
                    (np.float16(w[j * ws + k]) + 0.5)
                    * np.float16(rx[k]) * np.float16(ry[j])
                    + np.float16(mx[k]) + np.float16(my[j])
            )
        y[i * ys + k] = np.float16(y_local)

@cuda.jit
def kernel_fp16i8(B, N, M, x, w, mx, rx, my, ry, y):
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    k = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
    if i < B and k < M:
        y_local = 0.0
        for j in range(N):
            y_local += float(x[i, j]) * (
                    (float(w[j, k]) + 0.5)
                    * float(rx[k]) * float(ry[j, 0])
                    + float(mx[k]) + float(my[j, 0])
            )
        y[i, k] = np.float16(y_local)

@cuda.jit
def kernel_one_fp16i8(B, N, M, x, w, mx, rx, my, ry, y, xs, ws, ys):
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    k = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
    if i < B and k < M:
        y_local = 0.0
        for j in range(N):
            y_local += np.float16(x[i, j]) * (
                    (float(w[j, k]) + 0.5)
                    * np.float16(rx[k]) * np.float16(ry[j, 0])
                    + np.float16(mx[k]) + np.float16(my[j, 0])
            )
        y[i, k] = np.float16(y_local)

def wkv(B, T, C, w, u, k, v, y, aa, bb, pp, lens, numset):
    threads_per_block = min(C, 32)
    blocks_per_grid = B * C // threads_per_block
    k = k.reshape(-1)
    v = v.reshape(-1)
    aa = aa.reshape(-1)
    bb = bb.reshape(-1)
    pp = pp.reshape(-1)
    kernel_wkv[blocks_per_grid, threads_per_block](C, w, u, k, v, y, aa, bb, pp, lens, numset)
    y = y.reshape(T, C)
    aa = aa.reshape(B, C)
    bb = bb.reshape(B, C)
    pp = pp.reshape(B, C)

    return y, aa, bb, pp

def mm_seq(B, N, M, x, w, mx, rx, my, ry, y):
    threads_per_block = (1, 128)
    blocks_per_grid = ((B+threads_per_block[0]-1)//threads_per_block[0], (M+threads_per_block[1]-1)//threads_per_block[1])
    x = x.reshape(-1)
    w = w.reshape(-1)
    my = my.reshape(-1)
    ry = ry.reshape(-1)
    y = y.reshape(-1)
    kernel_mm_seq_fp16i8[blocks_per_grid, threads_per_block](B, N, M, x, w, mx, rx, my, ry, y, N, M, M)
    y = y.reshape(B, M)
    return y

def seq(B, N, M, x, w, mx, rx, my, ry, y):
    threads_per_block = (1, 128)
    blocks_per_grid = ((B+threads_per_block[0]-1)//threads_per_block[0], (M+threads_per_block[1]-1)//threads_per_block[1])
    kernel_fp16i8[blocks_per_grid, threads_per_block](B, N, M, x, w, mx, rx, my, ry, y)
    return y

def mmi8_one(B, N, M, x, w, mx, rx, my, ry, y):
    threads_per_block = (1, 1024)
    blocks_per_grid = ((B+threads_per_block[0]-1)//threads_per_block[0], (M+threads_per_block[1]-1)//threads_per_block[1])
    kernel_fp16i8[blocks_per_grid, threads_per_block](B, N, M, x, w, mx, rx, my, ry, y)
    return y

