# py-vowpal-wabbit

These are parallel bindings to the existing set of Python bindings for VW. These serve as a place to experiment with a new Python API as well as replace Boost Python with PyBind11 as the glue. The interfaces here are all subject to change so be prepared for things to change if you use this library. Note: parity with the existing bindings is not a goal of this project.

## Goals

## Local debug if using vcpkg deps
```
export CMAKE_TOOLCHAIN_FILE=$(pwd)/ext_libs/vcpkg/scripts/buildsystems/vcpkg.cmake
pip install -e .
pytest
```
