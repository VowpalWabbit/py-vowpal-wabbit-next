from typing import List, Optional
import typing
from vowpal_wabbit_next import _core, Example

PredictionType = _core.PredictionType
LabelType = _core.LabelType

Prediction = typing.Union[
    float,
    typing.List[float],
    typing.List[typing.Tuple[int, float]],
    typing.List[typing.List[typing.Tuple[int, float]]],
    int,
    typing.List[int],
    typing.List[typing.Tuple[float, float, float]],
    typing.Tuple[float, float],
    typing.Tuple[float, typing.List[int]],
    None,
]


class Workspace:
    def __init__(
        self,
        args: List[str],
        *,
        model_data: Optional[bytes] = None,
        _existing_workspace: Optional[_core.Workspace] = None
    ):
        if _existing_workspace is not None:
            self._workspace = _existing_workspace
        else:
            self._workspace = _core.Workspace(args, model_data=model_data)

    def predict_one(self, example: typing.Union[Example, List[Example]]) -> Prediction:
        if isinstance(example, Example):
            return self._workspace.predict_one(example)
        else:
            return self._workspace.predict_multi_ex_one(example)

    def learn_one(self, example: typing.Union[Example, List[Example]]) -> None:
        if isinstance(example, Example):
            self._workspace.learn_one(example)
        else:
            self._workspace.learn_multi_ex_one(example)

    @property
    def prediction_type(self) -> PredictionType:
        return self._workspace.get_prediction_type()

    @property
    def label_type(self) -> LabelType:
        return self._workspace.get_label_type()

    @property
    def multiline(self) -> bool:
        return self._workspace.get_is_multiline()

    def serialize(self) -> bytes:
        return self._workspace.serialize()

    def json_weights(
        self, *, include_feature_names: bool = False, include_online_state: bool = False
    ) -> str:
        return self._workspace.json_weights(
            include_feature_names=include_feature_names,
            include_online_state=include_online_state,
        )
