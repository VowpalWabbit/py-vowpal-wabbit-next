#!/usr/bin/env bash
set -e

pip install .
pip install pybind11-stubgen
pybind11-stubgen vowpal_wabbit_next._core --no-setup-py
mv stubs/vowpal_wabbit_next/_core-stubs/__init__.pyi src/vowpal_wabbit_next/_core.pyi
rm -r stubs
