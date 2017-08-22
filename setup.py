import codecs
import os
import re

from setuptools import find_packages, setup


# -------------------------------------------------------------

NAME = 'luminoth'
PACKAGES = find_packages()
META_PATH = os.path.join('luminoth', '__init__.py')
KEYWORDS = ['tensorflow', 'computer vision', 'object detection', 'rcnn', 'faster rcnn']
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
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
]
INSTALL_REQUIRES = [
    'numpy',
    'Pillow',
    'lxml',
    'click>=6.7,<7',
    'PyYAML>=3.12,<4',
    'easydict>=1.7,<2',
    'google-api-python-client>=1.6.2,<2',
    'google-cloud-storage>=1.2.0',
    'Flask>=0.12',
]
TEST_REQUIRES = []

# -------------------------------------------------------------


HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
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
# If TensorFlow or Sonnet are not installed, we might as well do that.
# Use the CPU versions by default. If the user wants to use the versions
# with GPU support, they must be installed in advance.
#
try:
    import tensorflow
except ImportError:
    INSTALL_REQUIRES += ['tensorflow>=1.2.1']

try:
    import sonnet
except ImportError:
    INSTALL_REQUIRES += ['dm-sonnet>=1.10']


setup(
    name=NAME,
    version=find_meta('version'),
    description=find_meta('description'),
    long_description=read('README.md'),
    license=find_meta('license'),
    author=find_meta('author'),
    maintainer=find_meta('author'),
    maintainer_email=find_meta('email'),
    url=find_meta('uri'),
    packages=PACKAGES,
    classifiers=CLASSIFIERS,
    package_data={'': ['*.yml']},
    setup_requires=[
    ],
    install_requires=INSTALL_REQUIRES,
    test_requires=TEST_REQUIRES,
    extras_require={
        'gpu support': [
            'tensorflow-gpu>=1.2.1',
            'dm-sonnet-gpu>=1.10',
        ],
    },
    entry_points="""
        [console_scripts]
        lumi=luminoth:cli
    """,
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*',
)
