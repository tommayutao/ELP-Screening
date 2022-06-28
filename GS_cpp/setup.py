import os
import re
import sys
from setuptools import setup, Extension

cpp_args = ['-std=c++11','-fopenmp']

ext_modules = [
	Extension(
	'GS_cpp',
		['src/compute_kernel.cpp','src/bindings.cpp'],
		include_dirs=['pybind11/include'],
	language='c++',
	extra_compile_args = cpp_args,
	extra_link_args=['-fopenmp']
	),
]

setup(
    name='GS_cpp',
    version='0.0.1',
    author='yutaoma',
    author_email='yma3@uchicago.edu',
    description='C++ implementation of gram matrix computation in GS kernel',
    ext_modules=ext_modules,
)
