import time

from setuptools import setup, find_packages

LONG_DESCRIPTION = open("README.md").read()
# VERSION = '0.1.0.%s' % int(time.time())
VERSION = '0.1.0'
setup(
    name='validity',
    version=VERSION,
    packages=find_packages(),
    long_description=LONG_DESCRIPTION,
    install_requires=[
        # 'fire>=0.4.0',
        # 'gast==0.4.0',
        # 'numpy==1.19.2',
        # 'scipy',
        # 'tensorflow==2.5.1',
        # 'tensorflow_addons',
        # 'tensorflow_datasets',
        # 'tensorflow_probability==0.13.0',
        # 'dm-sonnet',
        # 'luigi>=3.0'
        # 'tqdm',
        # 'typeguard',
        # 'Pillow',
        # 'submitit',
    ],
    extras_require={
        'tests': [
            # 'pylint',
            # 'pytest',
            # 'pytest-cov',
            # 'pytest-integration',
            # 'yapf',
        ],
    })
