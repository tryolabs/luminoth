from setuptools import find_packages, setup

setup(
    name='luminoth',
    description='Deep Learning toolkit',
    version='0.0.1.dev0',
    license='BSD 3-Clause License',
    packages=find_packages(),
    url="https://github.com/tryolabs/luminoth",
    include_package_data=True,
    package_data={'': ['*.yml']},
    setup_requires=[
    ],
    install_requires=[
        'numpy==1.13.1',
        'click==6.7',
        'Pillow==4.0.0',
        'PyYAML==3.12',
        'easydict==1.7',
        'lxml==3.8.0',
        'tensorflow==1.3',
        'dm-sonnet==1.10',
        'google-api-python-client==1.6.2',
        'google-cloud-storage==1.2.0',
        'Flask==0.12.2',
    ],
    entry_points="""
        [console_scripts]
        lumi=luminoth:cli
    """,
    python_requires='>=2.7',
    classifiers=(
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
    ),
)
