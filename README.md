# py-vowpal-wabbit-next

[![PyPI version](https://badge.fury.io/py/vowpal-wabbit-next.svg)](https://badge.fury.io/py/vowpal-wabbit-next)
[![Documentation Status](https://readthedocs.org/projects/vowpal-wabbit-next/badge/?version=latest)](https://vowpal-wabbit-next.readthedocs.io/en/latest/?badge=latest)

These are a new set of bindings for [VowpalWabbit](https://github.com/VowpalWabbit/vowpal_wabbit). Parity with the existing bindings is not a goal of this project as it is an opportunity for a clean slate and opportunity to rethink old decisions. The interfaces here are all subject to change so be prepared for things to change if you use this library.

## Installation

```sh
pip install vowpal-wabbit-next
```

## Example

```python
import vowpalwabbit_next as vw
import io

cb_input = io.StringIO(
    """shared | s
0:1:0.5 | a=0
| a=1

shared | s
| a=0
1:0:0.5 | a=1"""
)

workspace = vw.Workspace(["--cb_explore_adf"])

with vw.TextFormatReader(workspace, cb_input) as reader:
    for event in reader:
        print(workspace.predict_then_learn_one(event))
```

## Goals

- Fully typed and documented
- All artifacts automatically build in CI
- Focus on library usage instead of CLI behavior

## Supported platforms

Wheels are provided for the following platforms:

- OS+arch: Windows x86_x64, MacOS x86_x64, MacOS arm64, Linux x86_x64
- Python: 3.7 (except MacOS), 3.8, 3.9, 3.10, 3.11

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
