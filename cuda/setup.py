from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
    name='cuda_rwkv',
    version='1.0',
    include_dirs=['include'],
    ext_modules=[
        CppExtension(
            name='cuda_rwkv',
            sources=['cuda_rwkv.cpp', 'kernel_rwkv.cu']
            
        )
    ],
    cmdclass={
        'build_ext':BuildExtension
    }

)
