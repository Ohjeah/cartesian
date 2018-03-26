from pathlib import Path
from setuptools import find_packages, setup

CURRENT_DIR = Path(__file__).parent

NAME = "cartesian"
DESCRIPTION = "Minimal cartesian genetic programming for symbolic regression."
URL = "https://github.com/ohjeah/cartesian"
EMAIL = "info@markusqua.de"
AUTHOR = "Markus Quade"
PYTHON = ">=3.5"
LICENSE = "LGPL"
CLASSIFIERS = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    "Topic :: Scientific/Engineering :: Mathematics",
]


with open(CURRENT_DIR / "requirements.txt", "r") as f:
    REQUIRED = f.readlines()

with open(CURRENT_DIR / "README.md", "r", encoding="utf8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name=NAME,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=["test", "example"]),
    install_requires=REQUIRED,
    python_requires=PYTHON,
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    license=LICENSE,
    classifiers=CLASSIFIERS)
