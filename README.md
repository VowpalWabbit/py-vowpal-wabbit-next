# py-vowpalwabbit-next

These are parallel bindings to the existing set of Python bindings for VW. These serve as a place to experiment with a new Python API as well as replace Boost Python with PyBind11 as the glue. The interfaces here are all subject to change so be prepared for things to change if you use this library. Note: parity with the existing bindings is not a goal of this project.

## Goals

- Fully typed
- No command line based configuration
- ...

## Supported platforms

Wheels are provided for the following platforms:

- OS+arch: Windows x64, MacOS x64, MacOS arm64, Linux x64
- Python: 3.8, 3.9, 3.10

## Development information

### Local debug if using vcpkg deps
```sh
export CMAKE_TOOLCHAIN_FILE=$(pwd)/ext_libs/vcpkg/scripts/buildsystems/vcpkg.cmake
pip install -e .
pytest
```

### Update pybind11 module type stub

After updating the native module the type stub must be updated. This can be done automatically like so (ideally automatically in future, or at least checked if it is stale):
```sh
./generate_types.sh
```
