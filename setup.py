from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extra_compile_args = ["-O3"]
extensions = [
    Extension("phmm", ["phmm.pyx"],
              extra_compile_args=extra_compile_args
        ),
    Extension("hmm", ["hmm.pyx"],
              extra_compile_args=extra_compile_args
        ),
]

setup(
    ext_modules = cythonize(extensions)
)
