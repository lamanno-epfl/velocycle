from setuptools import setup, find_packages

setup(
    name='velocycle',
    version='0.0.1',
    packages=find_packages(),
    description='Bayesian model for RNA velocity estimation of periodic manifolds',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Alex Lederer',
    author_email='alex.lederer@epfl.ch',
    url='https://github.com/lamanno-epfl/velocycle'
)