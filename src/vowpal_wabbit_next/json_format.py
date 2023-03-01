import typing

from vowpal_wabbit_next import _core, Workspace, Example
from types import TracebackType


class JsonFormatParser:
    def __init__(self, workspace: Workspace):
        """Parse VW Json format examples.

        Args:
            workspace (Workspace): Workspace object used to configure this parser
        """
        self._workspace = workspace
        self._multi_line = workspace.multiline

    def parse_json(self, text: str) -> typing.Union[Example, typing.List[Example]]:
        '''Parse a single json object in json format.

        Examples:
            >>> from vowpal_wabbit_next import Workspace, JsonFormatParser
            >>> workspace = Workspace()
            >>> parser = JsonFormatParser(workspace)
            >>> json_str = """
            ... {
            ...     "_label": -1.0,
            ...     "feat1": 0.5,
            ...     "feat2": 2,
            ... }
            ... """
            >>> example = parser.parse_json(json_str)

        Args:
            text (str): JSON string of input

        Returns:
            typing.Union[Example, typing.List[Example]]: Parsed example or list of examples
        '''
        result = _core._parse_line_json(self._workspace._workspace, text)
        if self._multi_line:
            return [
                Example(_existing_example=ex, _label_type=self._workspace.label_type)
                for ex in result
            ]
        else:
            if len(result) != 1:
                raise ValueError("Expected single example")
            return Example(
                _existing_example=result[0], _label_type=self._workspace.label_type
            )


JsonFormatReaderT = typing.TypeVar("JsonFormatReaderT", bound="JsonFormatReader")


# takes a file and uses a context manager to generate based on the contents of the file
class JsonFormatReader:
    def __init__(self, workspace: Workspace, file: typing.TextIO):
        """Read VW Json format examples from the given text file. This reader always produces lists of examples.

        Examples:
            >>> from vowpal_wabbit_next import Workspace, JsonFormatReader
            >>> workspace = Workspace()
            >>> with open("data.txt", "r") as f:
            ...     with JsonFormatReader(workspace, f) as reader:
            ...         for example in reader:
            ...               workspace.predict_one(example)


        Args:
            workspace (Workspace): Workspace object used to configure this reader
            file (typing.BinaryIO): File to read from
        """
        self._parser = JsonFormatParser(workspace)
        self._workspace = workspace
        self._file = file
        if not self._workspace.multiline:
            raise ValueError("Must use a multiline Workspace for json format")

    def __enter__(self: JsonFormatReaderT) -> JsonFormatReaderT:
        return self

    def __exit__(
        self,
        exc_type: typing.Optional[typing.Type[BaseException]],
        exc_value: typing.Optional[BaseException],
        traceback: typing.Optional[TracebackType],
    ) -> None:
        self._file.close()

    def __iter__(self) -> typing.Iterator[typing.Union[Example, typing.List[Example]]]:
        if self._workspace.multiline:
            for line in self._file:
                yield self._parser.parse_json(line.rstrip())
