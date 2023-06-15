from vowpal_wabbit_next import _core
from vowpal_wabbit_next.labels import (
    LabelType,
    SimpleLabel,
    MulticlassLabel,
    CBLabel,
    CSLabel,
    CCBLabel,
)
from typing import List, Optional, Union, Iterator

AllLabels = Union[SimpleLabel, MulticlassLabel, CBLabel, CSLabel, CCBLabel, None]

FeatureGroupRef = _core.FeatureGroupRef


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

    def get_label(self) -> AllLabels:
        """Get the label of the example.

        Returns:
            AllLabels: The label of the example
        """
        return self._example._get_label(self.label_type)

    def set_label(self, label: AllLabels) -> None:
        """Set the label of the example.

        Args:
            label (AllLabels): The label to set

        Raises:
            ValueError: If the label type is not supported.
        """
        if isinstance(label, SimpleLabel):
            self.label_type = LabelType.Simple
        elif isinstance(label, MulticlassLabel):
            self.label_type = LabelType.Multiclass
        elif isinstance(label, CBLabel):
            self.label_type = LabelType.CB
        elif isinstance(label, CSLabel):
            self.label_type = LabelType.CS
        elif isinstance(label, CCBLabel):
            self.label_type = LabelType.CCB
        elif label is None:
            self.label_type = LabelType.NoLabel
        else:
            raise ValueError("Unsupported label type")

        self._example._set_label(label)

    @property
    def tag(self) -> str:
        """Get the tag of the example.

        Returns:
            str: The tag of the example
        """
        return self._example._get_tag()

    @tag.setter
    def tag(self, tag: str) -> None:
        """Set the tag of the example.

        Args:
            tag (str): The tag to set
        """
        self._example._set_tag(tag)

    @property
    def feat_group_indices(self) -> List[int]:
        """Get the populated feature groups for this example.

        Returns:
            List[int]: The populated feature groups for this example
        """
        return self._example._feat_group_indices

    def __iter__(self) -> Iterator[FeatureGroupRef]:
        """Iterate over the feature groups in this example.

        Examples:
            >>> for fg in example:
            >>>     print(fg.feat_group_index)
            >>>     print(fg.indices)
            >>>     print(fg.values)

        Yields:
            Iterator[FeatureGroupRef]: An iterator over the feature groups in this example
        """
        return self._example.__iter__()

    def __convert_key(self, key: Union[str, int]) -> int:
        if isinstance(key, str):
            return ord(key)
        else:
            # Key must be a valid byte
            if key < 0 or key >= 256:
                raise IndexError("Index out of range")
            return key

    def __getitem__(self, key: Union[str, int]) -> FeatureGroupRef:
        """Get a reference to a feature group. If it doesn't already exist it will be created.

        Args:
            key (Union[str, int]): Either the first character of namespace or index of namespace

        Examples:
            >>> print(example["d"].feat_group_index)
            >>> print(example["d"].indices)
            >>> print(example["d"].values)

        Returns:
            FeatureGroupRef: A reference to the feature group
        """
        return self._example.__getitem__(self.__convert_key(key))

    def __delitem__(self, key: Union[str, int]) -> None:
        """Delete a feature group.

        Args:
            key (Union[str, int]): Either the first character of namespace or index of namespace
        """
        return self._example.__delitem__(self.__convert_key(key))

    def __contains__(self, key: Union[str, int]) -> bool:
        """Check if a feature group exists.

        Args:
            key (Union[str, int]): Either the first character of namespace or index of namespace

        Returns:
            bool: True if the feature group exists, False otherwise
        """
        return self._example.__contains__(self.__convert_key(key))
