import sys

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages

setup(
    name="vowpal_wabbit_next",
    version="0.1.0",
    description="Experimental python bindings for VowpalWabbit",
    author="Jack Gerrits",
    license="BSD-3-Clause",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"vowpal_wabbit_next": ["py.typed", "_core.pyi"]},
    cmake_install_dir="src/vowpal_wabbit_next",
    install_requires=["numpy"],
    python_requires=">=3.8",
)
