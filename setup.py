from setuptools import setup, find_packages


setup(
    name='config-utils',
    version="0.0.1",
    packages=[package for package in find_packages() if package.startswith('config_utils')],
    install_requires=[
        'matplotlib',
        'numpy',
        'hydra-core',
        'memoization',
        'pytest',
    ],
    description='',
    author=''
)
