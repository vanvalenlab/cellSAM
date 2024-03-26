from setuptools import setup, find_packages
import sys

# Check Python version
if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or above is required.")

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='cellSAM',
    version='0.1.0',
    packages=find_packages(),
    package_data={
        'cellSAM': ['modelconfig.yaml'],
    },
    install_requires=requirements,  # Use the read requirements as install_requires
    python_requires='>=3.10',
)
