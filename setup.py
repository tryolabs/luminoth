from setuptools import find_packages, setup

setup(
    name='luminoth',
    description='Deep Learning toolkit',
    version='0.0.1',
    license='BSD 3-Clause License',
    packages=find_packages(),
    url="https://github.com/tryolabs/luminoth",
    setup_requires=[
    ],
    install_requires=[
        'numpy==1.13.1',
        'click==6.7',
        'Pillow==4.0.0',
        'PyYAML==3.12',
        'easydict==1.7',
        'lxml==3.8.0',
    ],
    entry_points="""
        [console_scripts]
        lumi=luminoth:cli
    """,
    python_requires='>=3.4',
)
