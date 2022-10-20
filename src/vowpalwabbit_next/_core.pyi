"""
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: vowpalwabbit_next

        .. autosummary::
           :toctree: _generate

           add
           subtract
    """
from __future__ import annotations
import vowpalwabbit_next._core
import typing

__all__ = ["add", "subtract"]

def add(arg0: int, arg1: int) -> int:
    """
    Add two numbers

    Some other explanation about the add function.
    """

def subtract(arg0: int, arg1: int) -> int:
    """
    Subtract two numbers

    Some other explanation about the subtract function.
    """

__version__ = "0.0.1"
