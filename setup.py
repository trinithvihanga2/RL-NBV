from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer_cuda',
    ext_modules=[
        CUDAExtension('chamfer_cuda', [
            'distance/chamfer_distance.cpp',
            'distance/chamfer_distance.cu',
        ], extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3']
        }),
        CUDAExtension('emd', [
            'distance/emd.cpp',
            'distance/emd_cuda.cu',
        ], extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3']
        })
    ],
    cmdclass={'build_ext': BuildExtension}
)