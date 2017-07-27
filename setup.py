import numpy as np

from Cython.Build import cythonize
from setuptools import find_packages, setup, Extension

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    Extension(
        "luminoth.utils.bbox",
        ["luminoth/utils/bbox.pyx"],
        include_dirs=[numpy_include]
    ),
]

setup(
    name='luminoth',
    description='Deep Learning toolkit',
    version='0.0.1',
    license='BSD 3-Clause License',
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    url="https://github.com/tryolabs/luminoth",
    setup_requires=[
        'numpy==1.13.1',
        'Cython==0.25.2',
    ],
    install_requires=[
        'click==6.7',
        'Pillow==4.0.0',
        'PyYAML==3.12',
        'easydict==1.7',
    ],
    entry_points="""
        [console_scripts]
        lumi=luminoth:cli
    """,
    python_requires='>=3.4',
)
