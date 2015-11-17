from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extra_compile_args = ["-O3"]
extensions = [
    Extension("aile.phmm", ["aile/phmm.pyx"],
              extra_compile_args=extra_compile_args
        ),
    Extension("aile.hmm", ["aile/hmm.pyx"],
              extra_compile_args=extra_compile_args
        ),
    Extension("aile._kernel", ["aile/_kernel.pyx"],
              extra_compile_args=extra_compile_args
        ),
    Extension("aile.dtw", ["aile/dtw.pyx"],
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
        'scikit-learn',
        'scrapely',
        'cython',
        'networkx',
        'pandas'],
    test_requires = [
        'pomegranate'
    ],
    ext_modules = cythonize(extensions)
)
