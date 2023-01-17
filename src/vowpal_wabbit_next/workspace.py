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
    _workspace: _core.Workspace

    def __init__(self, args: List[str], *, model_data: Optional[bytes] = None):
        self._workspace = _core.Workspace(args, model_data=model_data)

    def predict_one(self, example: typing.Union[Example, List[Example]]) -> Prediction:
        # TODO: ensure setup
        if isinstance(example, Example):
            return self._workspace.predict_one(example)
        else:
            return self._workspace.predict_multi_ex_one(example)

    def learn_one(self, example: typing.Union[Example, List[Example]]) -> None:
        # TODO: ensure setup
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

    def setup_example(
        self, example: typing.Union[Example, typing.List[Example]]
    ) -> None:
        if isinstance(example, Example):
            self._workspace.setup_example(example)
        else:
            for example in example:
                self._workspace.setup_example(example)

    # TODO: implement
    # def unsetup_example(self) -> None:
    #     # not implemented yet
    #     ...
