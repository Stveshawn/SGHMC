"""
Wed Apr 24 12:56:01 2019
@author: Steve shao
"""

from setuptools import setup

setup(
    name='sghmc',
    version='0.1',
    author='Steve Shao',
    author_email='lingyun.shao@duke.edu',
    scripts=['algs/sghmc.py'],
    license='LICENSE.txt',
    description='Implementation of SGHMC, HMC and SGLD algorithm',
    url = 'https://github.com/Stveshawn/SGHMC',
    keywords = ['Stochastic Gradient HMC', 'HMC', 'Stochastic Gradient MCMC'], 
    packages = ['sghmc'],
    install_requires=[
        "seaborn >= 0.7.0",
        "cppimport >= 18.1.10",
        "matplotlib >= 2.0.0",
        "pybind11 >= 2.2.2",
    ]
)