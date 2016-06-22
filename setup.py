import sys
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extra_compile_args = ["-O3"]
extensions = [
    Extension("aile._kernel", ["aile/_kernel.pyx"],
              extra_compile_args=extra_compile_args
        ),
    Extension("aile.dtw", ["aile/dtw.pyx"],
              extra_compile_args=extra_compile_args
    ),
]

setup_options = dict(
    name = 'AILE',
    version = '0.0.2',
    packages = ['aile'],
    install_requires = [
        'numpy',
        'scipy',
        'scikit-learn',
        'scrapely',
        'cython',
        'networkx',
        'pulp'
    ],
    dependency_links = [
        'git+https://github.com/scrapinghub/portia.git@multiple-item-extraction#egg=slyd&subdirectory=slyd',
        'git+https://github.com/scrapinghub/portia.git@multiple-item-extraction#egg=slybot&subdirectory=slybot'
    ],
    tests_requires = [
        'pytest'
    ],
    ext_modules = cythonize(extensions),
    scripts = ['scripts/gen-slybot-project']
)


# stolen from https://github.com/larsmans/seqlearn/blob/master/setup.py:
# For these actions, NumPy is not required. We want them to succeed without,
# for example when pip is used to install seqlearn without NumPy present.
NO_NUMPY_ACTIONS = ('--help-commands', 'egg_info', '--version', 'clean')
if not ('--help' in sys.argv[1:]
        or len(sys.argv) > 1 and sys.argv[1] in NO_NUMPY_ACTIONS):
    import numpy
    setup_options['include_dirs'] = [numpy.get_include()]


setup(**setup_options)
