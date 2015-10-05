from setuptools import setup
from Cython.Build import cythonize

setup(
    py_modules = ['phmm'],
    ext_modules = cythonize("hmm.pyx")
)
