pip install . --use-feature=in-tree-build
pip install pybind11-stubgen
pybind11-stubgen vowpalwabbit_next._core --no-setup-py
mv stubs/vowpalwabbit_next/_core-stubs/__init__.pyi src/vowpalwabbit_next/_core.pyi
rm -r stubs
