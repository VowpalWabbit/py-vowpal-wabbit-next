from __future__ import annotations
import vowpalwabbit_next._core
import typing

__all__ = [
    "Workspace"
]


class Workspace():
    def __init__(self, args: typing.List[str], *, model_data: typing.Optional[bytes] = None) -> None: ...
    pass
__version__ = '0.0.1'
