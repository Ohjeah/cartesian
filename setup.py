from setuptools import setup

here = os.path.abspath(dirname(__file__))

with open('README.rst') as f:
    long_description = '\n' + f.read()

with open("requirements.txt", "r") as f:
    required = f.readlines()

setup(
    name='cgp',
    version="0.0.0",
    description='Minimal cartesian genetic programming for symbolic regression.',
    long_description=long_description,
    author='Markus Quade',
    author_email='info@markusqua.de',
    url='https://github.com/ohjeah/cgp.py',
    packages=['cgp'],
    install_requires=required,
    license='MIT',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist bdist_wheel upload")
    sys.exit()
