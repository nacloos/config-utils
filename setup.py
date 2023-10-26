from setuptools import setup, find_packages


setup(
    name='config-utils',
    version="0.0.1",
    packages=[
        package for package in find_packages() if package.startswith('config_utils')
    ],
    package_data={
        "config_utils": ["cue_binaries/*"]
    }
    install_requires=[
        'matplotlib',
        'numpy',
        'hydra-core',
        'memoization',
        "requests",
        'pytest',
    ],
    description='',
    author=''
)
