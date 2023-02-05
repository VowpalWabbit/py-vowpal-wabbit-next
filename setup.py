import sys
import pathlib

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

with open(pathlib.Path(__file__).parent.resolve() / "README.md", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="vowpal-wabbit-next",
    version="0.1.0",
    description="Experimental python bindings for VowpalWabbit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="VowpalWabbit",
    license="BSD-3-Clause",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"vowpal_wabbit_next": ["py.typed", "_core.pyi"]},
    cmake_install_dir="src/vowpal_wabbit_next",
    install_requires=["numpy"],
    python_requires=">=3.7",
)
