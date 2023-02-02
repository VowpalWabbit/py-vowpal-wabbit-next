from typing import List, Optional
from vowpal_wabbit_next import _core, Workspace


class ModelDelta:
    def __init__(
        self, data: bytes, *, _existing_model_delta: Optional[_core.ModelDelta] = None
    ):
        """A delta between two VW models.

        The standard way to create one is with :py:meth:`vowpal_wabbit_next.calculate_delta`.

        Args:
            data (bytes): Bytes of a previously serialized ModelDelta to be used for loading
            _existing_model_delta (Optional[_core.ModelDelta], optional): This is an internal parameter and should not be used by end users.
        """
        if _existing_model_delta is not None:
            self._model_delta = _existing_model_delta
        else:
            self._model_delta = _core.ModelDelta(data)

    def serialize(self) -> bytes:
        """Serialize the delta.

        Returns:
            bytes: The serialized delta.
        """
        return self._model_delta.serialize()


def calculate_delta(base_model: Workspace, derived_model: Workspace) -> ModelDelta:
    """Produce a delta between two existing models.

    Args:
        base_model (Workspace): The base of the model
        derived_model (Workspace): The model produced from further training of base_model

    Returns:
        ModelDelta: The delta between the models.
    """
    return ModelDelta(
        bytes(),
        _existing_model_delta=_core._calculate_delta(
            base_model._workspace, derived_model._workspace
        ),
    )


def apply_delta(model: Workspace, delta: ModelDelta) -> Workspace:
    """Apply the delta to the model.

    Args:
        model (Workspace): The model to apply the delta to
        delta (ModelDelta): The delta to apply

    Returns:
        Workspace: The new model
    """
    return Workspace(
        [], _existing_workspace=_core._apply_delta(model._workspace, delta._model_delta)
    )


def merge_deltas(deltas: List[ModelDelta]) -> ModelDelta:
    """Merge a list of deltas into a single delta.

    Args:
        deltas (List[ModelDelta]): The deltas to merge. All deltas should come from the same base model.

    Returns:
        ModelDelta: The merged delta.
    """
    return ModelDelta(
        bytes(),
        _existing_model_delta=_core._merge_deltas([x._model_delta for x in deltas]),
    )
