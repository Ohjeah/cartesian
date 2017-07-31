import os
import sys
from setuptools import setup

with open('README.md') as f:
    long_description = '\n' + f.read()

with open("requirements.txt", "r") as f:
    required = f.readlines()

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist bdist_wheel upload")
    sys.exit()

setup(
    name='cartesian',
    version="0.1.0",
    description='Minimal cartesian genetic programming for symbolic regression.',
    long_description=long_description,
    author='Markus Quade',
    author_email='info@markusqua.de',
    url='https://github.com/ohjeah/cartesian',
    packages=['cartesian'],
    install_requires=required,
    license='MIT',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
