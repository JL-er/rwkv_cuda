#include <torch/extension.h>
#include "rwkv.h"
typedef at::Half fp16;

void cuda_wkv(int64_t B, int64_t C,
                torch::Tensor &w, torch::Tensor &u,
                torch::Tensor &k, torch::Tensor &v, torch::Tensor &y,
                torch::Tensor &aa, torch::Tensor &bb, torch::Tensor &pp,
                torch::Tensor &lens, torch::Tensor &numset){
                    wkv(B, C, 
                            w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<fp16>(), v.data_ptr<fp16>(), y.data_ptr<fp16>(), 
                            aa.data_ptr<float>(), bb.data_ptr<float>(), pp.data_ptr<float>(), lens.data_ptr<int>(), numset.data_ptr<int>());
                }

void cuda_i8seq(int64_t B, int64_t N, int64_t M,
                    torch::Tensor &x, torch::Tensor &w,
                    torch::Tensor &mx,
                    torch::Tensor &rx,
                    torch::Tensor &my,
                    torch::Tensor &ry,
                    torch::Tensor &y) {
                        i8seq(B, N, M,
                        x.data_ptr<fp16>(),
                        w.data_ptr<uint8_t>(),
                        mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
                        my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
                        y.data_ptr<fp16>());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("cuda_wkv", &cuda_wkv);
    m.def("cuda_i8seq", &cuda_i8seq);
}

TORCH_LIBRARY(rwkv, m) {
    m.def("cuda_wkv", cuda_wkv);
    m.def("cuda_i8seq", cuda_i8seq);
}