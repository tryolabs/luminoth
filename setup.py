import numpy as np

from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension


try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    Extension(
        "frcnn.utils.bbox",
        ["frcnn/utils/bbox.pyx"],
        include_dirs=[numpy_include]
    ),
]

setup(
  name='zresearch',
  ext_modules=cythonize(ext_modules),
)
