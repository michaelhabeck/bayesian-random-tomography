import os
import sys
import imp
import numpy

from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

try:
    __doc__ = open('README.md').read()
except IOError:
    pass

__file__ = './'
ROOT            = 'isdbeads'
LOCATION        = os.path.abspath(os.path.dirname(__file__))
JUNK            = []
NAME            = "isdbeads"
VERSION         = "0.1"
AUTHOR          = "Michael Habeck"
EMAIL           = "michael.habeck@uni-jena.de"
URL             = ""
SUMMARY         = "Inferential Structure Determination of Bead Models"
DESCRIPTION     = __doc__
LICENSE         = 'MIT'
REQUIRES        = ['numpy', 'scipy', 'csb', 'matplotlib']

module = Extension('isdbeads._isd',
                   define_macros = [('MAJOR_VERSION', '0'),
                                    ('MINOR_VERSION', '1'),
                                    ('PY_ARRAY_UNIQUE_SYMBOL','ISDBEADS')],
                      include_dirs = [numpy.get_include(), './isdbeads/c'],
                      extra_compile_args = ['-Wno-cpp'],
                      sources = ['./isdbeads/c/_isdmodule.c',
                                 './isdbeads/c/mathutils.c',
                                 './isdbeads/c/forcefield.c',
                                 './isdbeads/c/prolsq.c',
                                 './isdbeads/c/nblist.c',
                                 './isdbeads/c/rosetta.c', 
                                 ]
)

os.environ['CFLAGS'] = '-Wno-cpp'
setup(
    name=NAME,
    packages=find_packages(exclude=('tests', 'data', 'scripts')),
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    description=SUMMARY,
    long_description=DESCRIPTION,
    license=LICENSE,
    requires=REQUIRES,
    ext_modules=[module] + cythonize("isdbeads/*.pyx"), 
    include_dirs = [numpy.get_include(), './isdbeads/c'],
    cmdclass={'build_ext': build_ext},
    classifiers=(
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Software Development :: Libraries')
)

