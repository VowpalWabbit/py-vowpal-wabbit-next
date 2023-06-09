from __future__ import annotations
import vowpal_wabbit_next._core
import typing

__all__ = [
    "CBLabel",
    "CCBExampleType",
    "CCBLabel",
    "CSLabel",
    "DebugNode",
    "DenseParameters",
    "Example",
    "LabelType",
    "ModelDelta",
    "MulticlassLabel",
    "PredictionType",
    "SimpleLabel",
    "Workspace"
]


class CBLabel():
    def __init__(self, *, label: typing.Optional[typing.Union[typing.Tuple[float, float], typing.Tuple[int, float, float]]] = None, weight: float = 1.0, shared: bool = False) -> None: 
        """
        A label representing a contextual bandit problem.

        .. note::
          Currently the label can only contain 1 or 0 cb costs. There is a mode in vw for CB (non-adf) that allows for multiple cb_classes per example, but it is not currently accessible via direct label access. If creating examples/labels from parsed input it should still work as expected. If you need this feature, please open an issue on the github repo.

        Args:
          label (Optional[Union[Tuple[float, float], Tuple[int, float, float]]): This is (action, cost, probability). The same rules as VW apply for if the action can be left out of the tuple.
          weight (float): The weight of the example.
          shared (bool): Whether the example is shared. This is only used for ADF examples and must be the first example. There can only be one shared example per ADF example list.
        """
    def __repr__(self) -> str: ...
    @property
    def label(self) -> typing.Optional[typing.Tuple[int, float, float]]:
        """
            The label for the example. The format of the label is (action, cost, probability). If the action is not specified, it will be set to 0.

        :type: typing.Optional[typing.Tuple[int, float, float]]
        """
    @label.setter
    def label(self, arg1: typing.Optional[typing.Union[typing.Tuple[float, float], typing.Tuple[int, float, float]]]) -> None:
        """
        The label for the example. The format of the label is (action, cost, probability). If the action is not specified, it will be set to 0.
        """
    @property
    def shared(self) -> bool:
        """
            Whether the example is shared. This is only used for ADF examples and must be the first example. There can only be one shared example per ADF example list.

        :type: bool
        """
    @property
    def weight(self) -> float:
        """
            The weight of the example.

        :type: float
        """
    @weight.setter
    def weight(self, arg0: float) -> None:
        """
        The weight of the example.
        """
    pass
class CCBExampleType():
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
    Action: vowpal_wabbit_next._core.CCBExampleType # value = <CCBExampleType.Action: 2>
    Shared: vowpal_wabbit_next._core.CCBExampleType # value = <CCBExampleType.Shared: 1>
    Slot: vowpal_wabbit_next._core.CCBExampleType # value = <CCBExampleType.Slot: 3>
    Unset: vowpal_wabbit_next._core.CCBExampleType # value = <CCBExampleType.Unset: 0>
    __members__: dict # value = {'Unset': <CCBExampleType.Unset: 0>, 'Shared': <CCBExampleType.Shared: 1>, 'Action': <CCBExampleType.Action: 2>, 'Slot': <CCBExampleType.Slot: 3>}
    pass
class CCBLabel():
    def __init__(self, type: CCBExampleType, *, outcome: typing.Optional[typing.Tuple[float, typing.List[typing.Tuple[int, float]]]] = None, explicit_included_actions: typing.Optional[typing.List[int]] = None) -> None: 
        """
        A label representing a conditional contextual bandit problem.

        Args:
          type (CCBExampleType): The type of the example. Unset is invalid.
          outcome (Optional[Tuple[float, List[Tuple[int, float]]]]): The outcome of the example. The format of the outcome is (cost, [(action, probability)]).
          explicit_included_actions (Optional[List[int]]): The list of actions explicitly included from the slot.
        """
    def __repr__(self) -> str: ...
    @property
    def example_type(self) -> CCBExampleType:
        """
            Whether the example represents the shared context.

        :type: CCBExampleType
        """
    @property
    def explicit_included_actions(self) -> typing.Optional[typing.List[int]]:
        """
            The list of actions explicitly excluded from the slot.

        :type: typing.Optional[typing.List[int]]
        """
    @property
    def outcome(self) -> typing.Optional[typing.Tuple[float, typing.List[typing.Tuple[int, float]]]]:
        """
            The outcome of the example. The format of the outcome is (cost, [(action, probability)]).

        :type: typing.Optional[typing.Tuple[float, typing.List[typing.Tuple[int, float]]]]
        """
    pass
class CSLabel():
    def __init__(self, *, costs: typing.Optional[typing.List[typing.Tuple[float, float]]] = None, shared: bool = False) -> None: 
        """
        A label representing a cost sensitive classification problem.

        Args:
          costs (Optional[List[Tuple[int, float]]]): List of classes and costs. If there is no label, this should be None.
          shared (bool): Whether the example represents the shared context
        """
    def __repr__(self) -> str: ...
    @property
    def costs(self) -> typing.Optional[typing.List[typing.Tuple[int, float]]]:
        """
            The costs for the example. The format of the costs is (class_index, cost).

        :type: typing.Optional[typing.List[typing.Tuple[int, float]]]
        """
    @costs.setter
    def costs(self, arg1: typing.List[typing.Tuple[int, float]]) -> None:
        """
        The costs for the example. The format of the costs is (class_index, cost).
        """
    @property
    def shared(self) -> bool:
        """
            Whether the example represents the shared context.

        :type: bool
        """
    pass
class DebugNode():
    """
    A node in the computation tree of a single learn/predict call. This represents the state of the example as it is entering a given reduction.

    .. warning::
      This is a highly experimental feature.
    """
    @property
    def children(self) -> typing.List[DebugNode]:
        """
        The child computations that this node processed. This represents traversal of the stack.

        :type: typing.List[DebugNode]
        """
    @property
    def function(self) -> str:
        """
        The function that was called on this reduction. Either 'learn' or 'predict'.

        :type: str
        """
    @property
    def input_labels(self) -> typing.Union[typing.Union[SimpleLabel, MulticlassLabel, CBLabel, CSLabel, CCBLabel, None], typing.List[typing.Union[SimpleLabel, MulticlassLabel, CBLabel, CSLabel, CCBLabel, None]]]:
        """
        The label that was passed into this reduction. Or, list of labels if this reduction is a multi-example reduction.

        :type: typing.Union[typing.Union[SimpleLabel, MulticlassLabel, CBLabel, CSLabel, CCBLabel, None], typing.List[typing.Union[SimpleLabel, MulticlassLabel, CBLabel, CSLabel, CCBLabel, None]]]
        """
    @property
    def interactions(self) -> typing.Union[typing.List[str], typing.List[typing.List[str]]]:
        """
        The interactions that were used to generate the features for this reduction. Or, list of interactions if this reduction is a multi-example reduction.

        :type: typing.Union[typing.List[str], typing.List[typing.List[str]]]
        """
    @property
    def is_multiline(self) -> bool:
        """
        Whether this reduction is a multi-example reduction.

        :type: bool
        """
    @property
    def name(self) -> str:
        """
        Name of the reduction being called.

        :type: str
        """
    @property
    def num_examples(self) -> int:
        """
        The number of examples that were processed by this reduction. This is always 1 for single example reductions.

        :type: int
        """
    @property
    def offset(self) -> typing.Union[int, typing.List[int]]:
        """
        The offset of the example. Or, list of offsets if this reduction is a multi-example reduction. This also includes the stride of the bottom learner.

        :type: typing.Union[int, typing.List[int]]
        """
    @property
    def output_prediction(self) -> typing.Union[float, typing.List[float], typing.List[typing.Tuple[int, float]], typing.List[typing.List[typing.Tuple[int, float]]], int, typing.List[int], typing.List[typing.Tuple[float, float, float]], typing.Tuple[float, float], typing.Tuple[int, typing.List[int]], None]:
        """
        The prediction that this reduction produced.

        :type: typing.Union[float, typing.List[float], typing.List[typing.Tuple[int, float]], typing.List[typing.List[typing.Tuple[int, float]]], int, typing.List[int], typing.List[typing.Tuple[float, float, float]], typing.Tuple[float, float], typing.Tuple[int, typing.List[int]], None]
        """
    @property
    def partial_prediction(self) -> typing.Union[float, typing.List[float]]:
        """
        The partial prediction on the example after this reduction ran. Or, list of partial predictions if this reduction is a multi-example reduction. This is generally only set by the bottom of the stack.

        :type: typing.Union[float, typing.List[float]]
        """
    @property
    def self_duration_ns(self) -> int:
        """
        The duration of this reduction in nanoseconds. It does not include time it takes to call children.

        :type: int
        """
    @property
    def total_duration_ns(self) -> int:
        """
        The duration of this reduction and all children in nanoseconds.

        :type: int
        """
    @property
    def updated_prediction(self) -> typing.Union[float, typing.List[float]]:
        """
        The partial prediction on the example after this reduction ran. Or, list of partial predictions if this reduction is a multi-example reduction. This is generally only set by the bottom of the stack.

        :type: typing.Union[float, typing.List[float]]
        """
    @property
    def weight(self) -> typing.Union[float, typing.List[float]]:
        """
        The weight of the example. Or, list of weights if this reduction is a multi-example reduction.

        :type: typing.Union[float, typing.List[float]]
        """
    pass
class DenseParameters():
    pass
class Example():
    def __init__(self) -> None: ...
    def _get_label(self, arg0: LabelType) -> typing.Union[SimpleLabel, MulticlassLabel, CBLabel, CSLabel, CCBLabel, None]: ...
    def _get_tag(self) -> str: ...
    def _is_newline(self) -> bool: ...
    def _set_label(self, arg0: typing.Union[SimpleLabel, MulticlassLabel, CBLabel, CSLabel, CCBLabel, None]) -> None: ...
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
    CCB: vowpal_wabbit_next._core.LabelType # value = <LabelType.CCB: 7>
    CS: vowpal_wabbit_next._core.LabelType # value = <LabelType.CS: 4>
    Continuous: vowpal_wabbit_next._core.LabelType # value = <LabelType.Continuous: 10>
    Multiclass: vowpal_wabbit_next._core.LabelType # value = <LabelType.Multiclass: 6>
    Multilabel: vowpal_wabbit_next._core.LabelType # value = <LabelType.Multilabel: 5>
    NoLabel: vowpal_wabbit_next._core.LabelType # value = <LabelType.NoLabel: 9>
    Simple: vowpal_wabbit_next._core.LabelType # value = <LabelType.Simple: 0>
    Slates: vowpal_wabbit_next._core.LabelType # value = <LabelType.Slates: 8>
    __members__: dict # value = {'Simple': <LabelType.Simple: 0>, 'CB': <LabelType.CB: 1>, 'CBEval': <LabelType.CBEval: 2>, 'CS': <LabelType.CS: 4>, 'Multilabel': <LabelType.Multilabel: 5>, 'Multiclass': <LabelType.Multiclass: 6>, 'CCB': <LabelType.CCB: 7>, 'Slates': <LabelType.Slates: 8>, 'NoLabel': <LabelType.NoLabel: 9>, 'Continuous': <LabelType.Continuous: 10>}
    pass
class ModelDelta():
    def __init__(self, model_data: bytes) -> None: ...
    def serialize(self) -> bytes: ...
    pass
class MulticlassLabel():
    def __init__(self, label: int, weight: float = 1.0) -> None: 
        """
        A label representing a multiclass classification problem.

        Args:
          label (int): The label.
          weight (float): The weight of the example.
        """
    @property
    def label(self) -> int:
        """
            The class of this label.

        :type: int
        """
    @label.setter
    def label(self, arg0: int) -> None:
        """
        The class of this label.
        """
    @property
    def weight(self) -> float:
        """
            The weight of this label.

        :type: float
        """
    @weight.setter
    def weight(self, arg0: float) -> None:
        """
        The weight of this label.
        """
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
class SimpleLabel():
    def __init__(self, label: float, weight: float = 1.0, initial: float = 0.0) -> None: 
        """
        A label representing a simple regression problem.

        Args:
          label (float): The label.
          weight (float): The weight of the example.
          initial (float): The initial value of the prediction.
        """
    def __repr__(self) -> str: ...
    @property
    def initial(self) -> float:
        """
            The initial value of the prediction.

        :type: float
        """
    @initial.setter
    def initial(self, arg0: float) -> None:
        """
        The initial value of the prediction.
        """
    @property
    def label(self) -> float:
        """
            The label.

        :type: float
        """
    @label.setter
    def label(self, arg0: float) -> None:
        """
        The label.
        """
    @property
    def weight(self) -> float:
        """
            The weight of this label.

        :type: float
        """
    @weight.setter
    def weight(self, arg0: float) -> None:
        """
        The weight of this label.
        """
    pass
class Workspace():
    def __init__(self, args: typing.List[str], *, model_data: typing.Optional[bytes] = None, record_feature_names: bool = False, record_metrics: bool = False, debug: bool = False) -> None: ...
    def end_pass(self) -> None: ...
    def get_index_for_scalar_feature(self, feature_name: str, feature_value: typing.Optional[str] = None, namespace_name: str = ' ') -> int: ...
    def get_is_multiline(self) -> bool: ...
    def get_label_type(self) -> LabelType: ...
    def get_metrics(self) -> dict: ...
    def get_prediction_type(self) -> PredictionType: ...
    def json_weights(self, *, include_feature_names: bool = False, include_online_state: bool = False) -> str: ...
    def learn_multi_ex_one(self, examples: typing.List[Example]) -> typing.Union[None, typing.List[DebugNode]]: ...
    def learn_one(self, examples: Example) -> typing.Union[None, typing.List[DebugNode]]: ...
    def predict_multi_ex_one(self, examples: typing.List[Example]) -> typing.Union[typing.Union[float, typing.List[float], typing.List[typing.Tuple[int, float]], typing.List[typing.List[typing.Tuple[int, float]]], int, typing.List[int], typing.List[typing.Tuple[float, float, float]], typing.Tuple[float, float], typing.Tuple[int, typing.List[int]], None], typing.Tuple[typing.Union[float, typing.List[float], typing.List[typing.Tuple[int, float]], typing.List[typing.List[typing.Tuple[int, float]]], int, typing.List[int], typing.List[typing.Tuple[float, float, float]], typing.Tuple[float, float], typing.Tuple[int, typing.List[int]], None], DebugNode]]: ...
    def predict_one(self, examples: Example) -> typing.Union[typing.Union[float, typing.List[float], typing.List[typing.Tuple[int, float]], typing.List[typing.List[typing.Tuple[int, float]]], int, typing.List[int], typing.List[typing.Tuple[float, float, float]], typing.Tuple[float, float], typing.Tuple[int, typing.List[int]], None], typing.Tuple[typing.Union[float, typing.List[float], typing.List[typing.Tuple[int, float]], typing.List[typing.List[typing.Tuple[int, float]]], int, typing.List[int], typing.List[typing.Tuple[float, float, float]], typing.Tuple[float, float], typing.Tuple[int, typing.List[int]], None], DebugNode]]: ...
    def predict_then_learn_multi_ex_one(self, examples: typing.List[Example]) -> typing.Union[typing.Union[float, typing.List[float], typing.List[typing.Tuple[int, float]], typing.List[typing.List[typing.Tuple[int, float]]], int, typing.List[int], typing.List[typing.Tuple[float, float, float]], typing.Tuple[float, float], typing.Tuple[int, typing.List[int]], None], typing.Tuple[typing.Union[float, typing.List[float], typing.List[typing.Tuple[int, float]], typing.List[typing.List[typing.Tuple[int, float]]], int, typing.List[int], typing.List[typing.Tuple[float, float, float]], typing.Tuple[float, float], typing.Tuple[int, typing.List[int]], None], typing.List[DebugNode]]]: ...
    def predict_then_learn_one(self, examples: Example) -> typing.Union[typing.Union[float, typing.List[float], typing.List[typing.Tuple[int, float]], typing.List[typing.List[typing.Tuple[int, float]]], int, typing.List[int], typing.List[typing.Tuple[float, float, float]], typing.Tuple[float, float], typing.Tuple[int, typing.List[int]], None], typing.Tuple[typing.Union[float, typing.List[float], typing.List[typing.Tuple[int, float]], typing.List[typing.List[typing.Tuple[int, float]]], int, typing.List[int], typing.List[typing.Tuple[float, float, float]], typing.Tuple[float, float], typing.Tuple[int, typing.List[int]], None], typing.List[DebugNode]]]: ...
    def readable_model(self, *, include_feature_names: bool = False) -> str: ...
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
def _parse_line_json(workspace: Workspace, line: str) -> typing.List[Example]:
    pass
def _parse_line_text(workspace: Workspace, line: str) -> Example:
    pass
def _run_cli_driver(args: typing.List[str], *, onethread: bool = False) -> typing.Tuple[typing.Optional[str], str, typing.List[str]]:
    pass
def _write_cache_example(workspace: Workspace, example: Example, file: object) -> None:
    pass
def _write_cache_header(workspace: Workspace, file: object) -> None:
    pass
__version__ = '0.4.1'
_vw_commit = 'b8c4ee3'
_vw_version = '9.8.0'
