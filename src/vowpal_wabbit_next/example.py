from vowpal_wabbit_next import _core
from vowpal_wabbit_next.labels import LabelType, SimpleLabel
from typing import Optional, Union


class Example:
    def __init__(
        self,
        *,
        _existing_example: Optional[_core.Example] = None,
        _label_type: LabelType = LabelType.NoLabel
    ):
        """A single VW example.

        Args:
            _existing_example (Optional[_core.Example], optional): This is an internal parameter and should not be used by end users.
            _label_type (LabelType, optional): This is an internal parameter and should not be used by end users.
        """
        if _existing_example is not None:
            if _label_type is None:
                raise ValueError(
                    "Must provide label type when using an existing example"
                )
            self._example = _existing_example
            self.label_type = _label_type
        else:
            self._example = _core.Example()
            self.label_type = LabelType.NoLabel

    def get_label(self) -> Union[SimpleLabel, None]:
        """Get the label of the example.

        Returns:
            Union[SimpleLabel, None]: The label of the example
        """
        return self._example._get_label(self.label_type)

    def set_label(self, label: Union[SimpleLabel, None]) -> None:
        """Set the label of the example.

        Args:
            label (Union[SimpleLabel, None]): The label to set

        Raises:
            ValueError: If the label type is not supported.
        """
        if isinstance(label, SimpleLabel):
            self.label_type = LabelType.Simple
        elif label is None:
            self.label_type = LabelType.NoLabel
        else:
            raise ValueError("Unsupported label type")

        self._example._set_label(label)
