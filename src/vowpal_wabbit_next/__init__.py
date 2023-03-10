from ._core import __version__, _vw_version, _vw_commit
from .example import Example
from .workspace import Workspace, DebugNode
from .text_format import TextFormatParser, TextFormatReader
from .json_format import JsonFormatParser, JsonFormatReader
from .dsjson_format import DSJsonFormatParser, DSJsonFormatReader
from .cache_format import CacheFormatWriter, CacheFormatReader
from .delta import ModelDelta, calculate_delta, apply_delta, merge_deltas
from .cli_driver import CLIError, run_cli_driver
from .prediction_type import PredictionType
from .labels import (
    LabelType,
    SimpleLabel,
    MulticlassLabel,
    CBLabel,
    CSLabel,
    CCBLabel,
    CCBExampleType,
)


VW_COMMIT: str = _vw_commit
"""Commit of VowpalWabbit that this package is built with"""
VW_VERSION: str = _vw_version
"""Version number of VowpalWabbit that this package is built with"""

__all__ = [
    "__version__",
    "apply_delta",
    "CacheFormatReader",
    "CacheFormatWriter",
    "calculate_delta",
    "CBLabel",
    "CLIError",
    "CSLabel",
    "CCBExampleType",
    "CCBLabel",
    "DebugNode",
    "DSJsonFormatParser",
    "DSJsonFormatReader",
    "Example",
    "JsonFormatParser",
    "JsonFormatReader",
    "LabelType",
    "merge_deltas",
    "ModelDelta",
    "MulticlassLabel",
    "PredictionType",
    "run_cli_driver",
    "SimpleLabel",
    "TextFormatParser",
    "TextFormatReader",
    "VW_COMMIT",
    "VW_VERSION",
    "Workspace",
]
