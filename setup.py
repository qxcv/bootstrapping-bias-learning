#!/usr/bin/env python3
import os

from setuptools import find_packages, setup


def readme():
    """Load README for use as package's long description."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(this_dir, "README.md"), "r") as fp:
        return fp.read()


setup(
    name="pref-bootstrap",
    version="0.0.1alpha1",
    python_requires=">=3.6",
    description="Bootstrapping preference models from rationality priors "
    "(EE227BT project)",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("old",)),
    install_requires=[
        "jupyterlab>=2.2.6",
        "numpy>=1.19.0",
        "scipy>=1.5.1",
        "seaborn>=0.10.1",
        "pandas>=1.0.5",
        "matplotlib>=3.3.0",
        "jax>=0.2.5",
        "jaxlib>=0.1.56",
        "gym>=0.17.3",
        # these are really dev/test dependencies rather than package deps (if
        # we ever publish this then we can put them in a 'dev' or 'test' extra)
        "ipdb",
        "ipython",
        "isort~=5.0",
        "black",
        "flake8",
        "pytest",
        "pytype",
    ],
)
