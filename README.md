# py-vowpal-wabbit-next

[![Documentation Status](https://readthedocs.org/projects/vowpal-wabbit-next/badge/?version=latest)](https://vowpal-wabbit-next.readthedocs.io/en/latest/?badge=latest)


These are a new set of bindings for VowpalWabbit. Parity with the existing bindings is not a goal of this project as it is an opportunity for a clean slate and opportunity to rethink old decisions. The interfaces here are all subject to change so be prepared for things to change if you use this library.

## Installation

```sh
pip install vowpal-wabbit-next
```

## Goals

- Fully typed and documented
- All artifacts automatically build in CI
- ...

## Supported platforms

Wheels are provided for the following platforms:

- OS+arch: Windows x64, MacOS x64, MacOS arm64, Linux x64
- Python: 3.7 (except MacOS arm64), 3.8, 3.9, 3.10, 3.11

## Development information

### Local debug if using vcpkg deps
```sh
# Build
export CMAKE_TOOLCHAIN_FILE=$(pwd)/ext_libs/vcpkg/scripts/buildsystems/vcpkg.cmake
pip install -v .

# Install dev dependncies
pip install -r requirements-dev.txt

# Test
pytest

# Type check
mypy

# Format
black src tests

# Check documentation
pydocstyle src
```

### Update pybind11 module type stub

After updating the native module the type stub must be updated. This can be done automatically like so (ideally automatically in future, or at least checked if it is stale):
```sh
./generate_types.sh
```
