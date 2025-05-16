from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="custom_cuda_allocator",
    ext_modules=[
        CUDAExtension(
            name="custom_cuda_allocator",
            sources=["custom_cuda_allocator.cpp"],
            extra_compile_args={
                'cxx': ['-O3'],     # C++17 is implied
                'nvcc': ['-O3'],    # NVCC flags
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)