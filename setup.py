import os

import versioneer
from setuptools import find_packages, setup

NAME = "cartesian"
DESCRIPTION = "Minimal cartesian genetic programming for symbolic regression."
URL = "https://github.com/ohjeah/cartesian"
EMAIL = "info@markusqua.de"
AUTHOR = "Markus Quade"

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt"), "r") as f:
    REQUIRED = f.readlines()

setup(
    name=NAME,
    version=versioneer.get_version(),
    description=DESCRIPTION,
    long_description=__doc__,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=["test", "example"]),
    install_requires=REQUIRED,
    license="MIT",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    cmdclass=versioneer.get_cmdclass(),
)
