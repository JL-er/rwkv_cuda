#include <iostream>
#include <math.h>
#include <cuda_fp16.h>
#include "ATen/ATen.h"
typedef at::Half fp16;


template<typename F>
__global__
void kernel_wkv(const int C,
                const float *_w, const float *_u, const F *_k, const F *_v,
                F *_y, float * _aa, float *_bb, float *_pp, const int *lens, const int *numset){
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    const int _b = idx / C;
                    const int _c = idx % C;
                    int _b_offset = 0;
                    if ((_b-1) >= 0){
                        _b_offset = numset[_b-1]*C;
                    }
                    int _offset = _b_offset;
                    int _state_offset = _b * C + _c;
                    int t = lens[_b];
                    float u = _u[_c];
                    float w = _w[_c];
                    const F *k = _k;
                    const F *v = _v;
                    F *y = _y;
                    float aa = _aa[_state_offset];
                    float bb = _bb[_state_offset];
                    float pp = _pp[_state_offset];
                    for(int i=0; i<t; ++i){
                        int ii = i * C + _offset + _c;
                        float kk = float(k[ii]);
                        float vv = float(v[ii]);
                        float ww = u + kk;
                        float p = max(pp, ww);
                        float e1 = exp(pp - p);
                        float e2 = exp(ww - p);
                        y[ii] = F((e1 * aa + e2 * vv) / (e1 * bb + e2));
                        ww = w + pp;
                        p = max(ww, kk);
                        e1 = exp(ww - p);
                        e2 = exp(kk - p);
                        aa = e1 * aa + e2 * vv;
                        bb = e1 * bb + e2;
                        pp = p;
                    }

                    _aa[_state_offset] = aa;
                    _bb[_state_offset] = bb;
                    _pp[_state_offset] = pp;
                }


template<typename F>
void wkv(int B, int C, float *w, float *u, F *k, F *v, F *y, float *aa, float *bb, float *pp, int *lens, int *numset) {
    dim3 block( min(C, 32) );
    assert(B * C % block.x == 0);
    dim3 grid(B * C / block.x);
    kernel_wkv<<<grid, block>>>(C, w, u, k, v, y, aa, bb, pp, lens, numset);
}

template void wkv<fp16>(
    int B, int C,
    float *w, float *u, fp16 *k, fp16 *v, fp16 *y,
    float *aa, float *bb, float *pp, int *lens, int *numset);

template void wkv<float>(
    int B, int C,
    float *w, float *u, float *k, float *v, float *y,
    float *aa, float *bb, float *pp, int *lens, int *numset);

__global__ void kernel_i8seq(const int B,const int N, const int M, 
                                const  fp16 *x, const uint8_t *w,
                                const fp16 *mx,
                                const fp16 *rx,
                                const fp16 *my,
                                const fp16 *ry,
                                fp16 *y){
                                    const int i = blockIdx.x * blockDim.x + threadIdx.x;
                                    const int k = blockIdx.y * blockDim.y + threadIdx.y;

                                    if (i < B && k < M) {
                                        float y_local = 0;
                                        for (int j = 0; j < N; ++j) {
                                            y_local += __half2float(x[i * N + j]) * (
                                                (float(w[j * M + k]) + 0.5f)
                                                * __half2float(rx[k]) * __half2float(ry[j]) + __half2float(mx[k]) + __half2float(my[j])
                                            );
                                        }
                                        y[i * M + k] = __float2half(y_local);
                                    }
                                }

void i8seq( int B, int N,  int M, 
                fp16 *x, uint8_t *w,
                fp16 *mx,
                fp16 *rx,
                fp16 *my,
                fp16 *ry,
                fp16 *y){
                    dim3 block(1, 128);
                    dim3 grid((B+block.x-1)/block.x, (M+block.y-1)/block.y);
                    //kernel_i8seq<<<grid, block>>>(B,N,M,cast(x),w,cast(mx),cast(rx),cast(my),cast(ry),cast(y));
                    kernel_i8seq<<<grid, block>>>(B,N,M,x,w,mx,rx,my,ry,y);
                }