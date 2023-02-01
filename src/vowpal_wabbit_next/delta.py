from typing import List, Optional
from vowpal_wabbit_next import _core, Workspace


class ModelDelta:
    def __init__(
        self, data: bytes, *, _existing_model_delta: Optional[_core.ModelDelta] = None
    ):
        if _existing_model_delta is not None:
            self._model_delta = _existing_model_delta
        else:
            self._model_delta = _core.ModelDelta(data)

    def serialize(self) -> bytes:
        return self._model_delta.serialize()


def calculate_delta(base_model: Workspace, derived_model: Workspace) -> ModelDelta:
    return ModelDelta(
        bytes(),
        _existing_model_delta=_core._calculate_delta(
            base_model._workspace, derived_model._workspace
        ),
    )


def apply_delta(model: Workspace, delta: ModelDelta) -> Workspace:
    return Workspace(
        [], _existing_workspace=_core._apply_delta(model._workspace, delta._model_delta)
    )


def merge_deltas(deltas: List[ModelDelta]) -> ModelDelta:
    return ModelDelta(
        bytes(),
        _existing_model_delta=_core._merge_deltas([x._model_delta for x in deltas]),
    )
