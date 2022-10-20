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
    name="vowpalwabbit_next",
    version="0.0.1",
    description="Experimental python bindings for VowpalWabbit",
    author="Jack Gerrits",
    license="BSD-3-Clause",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"vowpalwabbit_next": ["py.typed", "_core.pyi"]},
    cmake_install_dir="src/vowpalwabbit_next",
    extras_require={"test": ["pytest", "mypy"], "docs": ["furo", "sphinx"]},
    python_requires=">=3.8",
)
