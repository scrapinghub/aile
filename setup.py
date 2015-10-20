from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extra_compile_args = ["-O3"]
extensions = [
    Extension("phmm", ["aile/phmm.pyx"],
              extra_compile_args=extra_compile_args
        ),
    Extension("hmm", ["aile/hmm.pyx"],
              extra_compile_args=extra_compile_args
        ),
]

setup(
    name = 'AILE',
    version = '0.0.1',
    packages = ['aile'],
    install_requires = [
        'numpy',
        'scipy',
        'scrapely',
        'tabulate',
        'selenium',
        'pomegranate',
        'cython',
        'pandas'],
    ext_modules = cythonize(extensions)
)
