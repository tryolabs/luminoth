import codecs
import os
import re

from setuptools import find_packages, setup


# -------------------------------------------------------------

NAME = 'luminoth'
PACKAGES = find_packages()
META_PATH = os.path.join('luminoth', '__init__.py')
KEYWORDS = [
    'tensorflow', 'computer vision', 'object detection', 'toolkit', 'deep learning',
    'faster rcnn'
]
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Natural Language :: English',
    'License :: OSI Approved :: BSD License',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Software Development :: Libraries',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
]

MIN_TENSORFLOW_VERSION = '1.3.0'

INSTALL_REQUIRES = [
    'numpy',
    'Pillow',
    'lxml',
    'dm-sonnet>=1.12',
    'click>=6.7,<7',
    'PyYAML>=3.12,<4',
    'easydict>=1.7,<2',
    'google-api-python-client>=1.6.2,<2',
    'google-cloud-storage>=1.2.0',
    'Flask>=0.12',
    'six>=1.11',
]
TEST_REQUIRES = []

# -------------------------------------------------------------


HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), 'rb', 'utf-8') as f:
        return f.read()


META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError('Unable to find __{meta}__ string.'.format(meta=meta))


#
# If TensorFlow is not installed, install it using the CPU version by default.
# If the user wants to use the version with GPU support, it must be installed
# in advance.
#
try:
    import tensorflow
except ImportError:
    INSTALL_REQUIRES += ['tensorflow>={}'.format(MIN_TENSORFLOW_VERSION)]


setup(
    name=NAME,
    version=find_meta('version'),
    description=find_meta('description'),
    long_description=read('README.md'),
    license=find_meta('license'),
    author=find_meta('author'),
    author_email=find_meta('email'),
    maintainer=find_meta('author'),
    maintainer_email=find_meta('email'),
    url=find_meta('uri'),
    packages=PACKAGES,
    classifiers=CLASSIFIERS,
    include_package_data=True,
    setup_requires=[
    ],
    install_requires=INSTALL_REQUIRES,
    test_requires=TEST_REQUIRES,
    extras_require={
        'gpu support': [
            'tensorflow-gpu>={}'.format(MIN_TENSORFLOW_VERSION),
        ],
    },
    entry_points="""
        [console_scripts]
        lumi=luminoth:cli
    """,
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*',
)
