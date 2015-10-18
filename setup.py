from setuptools import setup
from Cython.Build import cythonize

modules = cythonize(["hmm.pyx", "phmm.pyx"])
setup(
    ext_modules = modules
)

for m in modules:
    m.extra_compile_args.extend(["-O3", "-msse", "-msse2", "-msse3"])
