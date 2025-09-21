import os
from setuptools import setup, find_packages

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="MeTr",
    packages=find_packages(),
    version="1.0.0",
    description="Me_Trans (Variational Autoencoder for the integration of Metabolomic and Transcriptomic Profiles)",
    author="TBH2025 Team 25",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.11',
)