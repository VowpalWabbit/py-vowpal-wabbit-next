from __future__ import annotations
import vowpal_wabbit_next._core
import typing

__all__ = [
    "DenseParameters",
    "Example",
    "LabelType",
    "ModelDelta",
    "PredictionType",
    "Workspace"
]


class DenseParameters():
    pass
class Example():
    def __init__(self) -> None: ...
    def _is_newline(self) -> bool: ...
    pass
class LabelType():
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    CB: vowpal_wabbit_next._core.LabelType # value = <LabelType.CB: 1>
    CBEval: vowpal_wabbit_next._core.LabelType # value = <LabelType.CBEval: 2>
    CCB: vowpal_wabbit_next._core.LabelType # value = <LabelType.CCB: 6>
    CS: vowpal_wabbit_next._core.LabelType # value = <LabelType.CS: 3>
    Continuous: vowpal_wabbit_next._core.LabelType # value = <LabelType.Continuous: 9>
    Multiclass: vowpal_wabbit_next._core.LabelType # value = <LabelType.Multiclass: 5>
    Multilabel: vowpal_wabbit_next._core.LabelType # value = <LabelType.Multilabel: 4>
    NoLabel: vowpal_wabbit_next._core.LabelType # value = <LabelType.NoLabel: 8>
    Simple: vowpal_wabbit_next._core.LabelType # value = <LabelType.Simple: 0>
    Slates: vowpal_wabbit_next._core.LabelType # value = <LabelType.Slates: 7>
    __members__: dict # value = {'Simple': <LabelType.Simple: 0>, 'CB': <LabelType.CB: 1>, 'CBEval': <LabelType.CBEval: 2>, 'CS': <LabelType.CS: 3>, 'Multilabel': <LabelType.Multilabel: 4>, 'Multiclass': <LabelType.Multiclass: 5>, 'CCB': <LabelType.CCB: 6>, 'Slates': <LabelType.Slates: 7>, 'NoLabel': <LabelType.NoLabel: 8>, 'Continuous': <LabelType.Continuous: 9>}
    pass
class ModelDelta():
    def __init__(self, model_data: bytes) -> None: ...
    def serialize(self) -> bytes: ...
    pass
class PredictionType():
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    ActionPdfValue: vowpal_wabbit_next._core.PredictionType # value = <PredictionType.ActionPdfValue: 10>
    ActionProbs: vowpal_wabbit_next._core.PredictionType # value = <PredictionType.ActionProbs: 4>
    ActionScores: vowpal_wabbit_next._core.PredictionType # value = <PredictionType.ActionScores: 2>
    ActiveMulticlass: vowpal_wabbit_next._core.PredictionType # value = <PredictionType.ActiveMulticlass: 11>
    DecisionProbs: vowpal_wabbit_next._core.PredictionType # value = <PredictionType.DecisionProbs: 9>
    Multiclass: vowpal_wabbit_next._core.PredictionType # value = <PredictionType.Multiclass: 5>
    MulticlassProbs: vowpal_wabbit_next._core.PredictionType # value = <PredictionType.MulticlassProbs: 8>
    Multilabels: vowpal_wabbit_next._core.PredictionType # value = <PredictionType.Multilabels: 6>
    NoPred: vowpal_wabbit_next._core.PredictionType # value = <PredictionType.NoPred: 12>
    Pdf: vowpal_wabbit_next._core.PredictionType # value = <PredictionType.Pdf: 3>
    Prob: vowpal_wabbit_next._core.PredictionType # value = <PredictionType.Prob: 7>
    Scalar: vowpal_wabbit_next._core.PredictionType # value = <PredictionType.Scalar: 0>
    Scalars: vowpal_wabbit_next._core.PredictionType # value = <PredictionType.Scalars: 1>
    __members__: dict # value = {'Scalar': <PredictionType.Scalar: 0>, 'Scalars': <PredictionType.Scalars: 1>, 'ActionScores': <PredictionType.ActionScores: 2>, 'Pdf': <PredictionType.Pdf: 3>, 'ActionProbs': <PredictionType.ActionProbs: 4>, 'Multiclass': <PredictionType.Multiclass: 5>, 'Multilabels': <PredictionType.Multilabels: 6>, 'Prob': <PredictionType.Prob: 7>, 'MulticlassProbs': <PredictionType.MulticlassProbs: 8>, 'DecisionProbs': <PredictionType.DecisionProbs: 9>, 'ActionPdfValue': <PredictionType.ActionPdfValue: 10>, 'ActiveMulticlass': <PredictionType.ActiveMulticlass: 11>, 'NoPred': <PredictionType.NoPred: 12>}
    pass
class Workspace():
    def __init__(self, args: typing.List[str], *, model_data: typing.Optional[bytes] = None) -> None: ...
    def get_index_for_scalar_feature(self, feature_name: str, feature_value: typing.Optional[str] = None, namespace_name: str = ' ') -> int: ...
    def get_is_multiline(self) -> bool: ...
    def get_label_type(self) -> LabelType: ...
    def get_prediction_type(self) -> PredictionType: ...
    def json_weights(self, *, include_feature_names: bool = False, include_online_state: bool = False) -> str: ...
    def learn_multi_ex_one(self, examples: typing.List[Example]) -> None: ...
    def learn_one(self, examples: Example) -> None: ...
    def predict_multi_ex_one(self, examples: typing.List[Example]) -> typing.Union[float, typing.List[float], typing.List[typing.Tuple[int, float]], typing.List[typing.List[typing.Tuple[int, float]]], int, typing.List[int], typing.List[typing.Tuple[float, float, float]], typing.Tuple[float, float], typing.Tuple[int, typing.List[int]], None]: ...
    def predict_one(self, examples: Example) -> typing.Union[float, typing.List[float], typing.List[typing.Tuple[int, float]], typing.List[typing.List[typing.Tuple[int, float]]], int, typing.List[int], typing.List[typing.Tuple[float, float, float]], typing.Tuple[float, float], typing.Tuple[int, typing.List[int]], None]: ...
    def predict_then_learn_multi_ex_one(self, examples: typing.List[Example]) -> typing.Union[float, typing.List[float], typing.List[typing.Tuple[int, float]], typing.List[typing.List[typing.Tuple[int, float]]], int, typing.List[int], typing.List[typing.Tuple[float, float, float]], typing.Tuple[float, float], typing.Tuple[int, typing.List[int]], None]: ...
    def predict_then_learn_one(self, examples: Example) -> typing.Union[float, typing.List[float], typing.List[typing.Tuple[int, float]], typing.List[typing.List[typing.Tuple[int, float]]], int, typing.List[int], typing.List[typing.Tuple[float, float, float]], typing.Tuple[float, float], typing.Tuple[int, typing.List[int]], None]: ...
    def serialize(self) -> bytes: ...
    def weights(self) -> DenseParameters: ...
    pass
class _CacheReader():
    def __init__(self, arg0: Workspace, arg1: object) -> None: ...
    def _get_next(self) -> typing.Optional[Example]: ...
    pass
def _apply_delta(base_workspace: Workspace, delta: ModelDelta) -> Workspace:
    pass
def _calculate_delta(base_workspace: Workspace, derived_workspace: Workspace) -> ModelDelta:
    pass
def _merge_deltas(deltas: typing.List[ModelDelta]) -> ModelDelta:
    pass
def _parse_line_dsjson(workspace: Workspace, line: str) -> typing.List[Example]:
    pass
def _parse_line_text(workspace: Workspace, line: str) -> Example:
    pass
def _run_cli_driver(args: typing.List[str], *, onethread: bool = False) -> typing.Tuple[typing.Optional[str], str, typing.List[str]]:
    pass
def _write_cache_example(workspace: Workspace, example: Example, file: object) -> None:
    pass
def _write_cache_header(workspace: Workspace, file: object) -> None:
    pass
__version__ = '0.0.1'
_vw_commit = 'dcbcd07'
_vw_version = '9.7.0'
