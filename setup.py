from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name='velocycle',
    version='0.1.0.2',
    packages=find_packages(),
    description='Bayesian model for RNA velocity estimation of periodic manifolds',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    author='@lamanno-epfl',
    author_email='gioele.lamanno@epfl.ch',
    url='https://github.com/lamanno-epfl/velocycle'
)
