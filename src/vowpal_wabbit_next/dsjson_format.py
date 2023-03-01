import typing

from vowpal_wabbit_next import _core, Workspace, Example
from types import TracebackType


class DSJsonFormatParser:
    def __init__(self, workspace: Workspace):
        """Parse VW DSJson format examples.

        Args:
            workspace (Workspace): Workspace object used to configure this parser
        """
        self._workspace = workspace

    def parse_json(self, text: str) -> typing.List[Example]:
        '''Parse a single json object in dsjson format.

        Examples:
            >>> from vowpal_wabbit_next import Workspace, DSJsonFormatParser
            >>> workspace = Workspace(["--cb_explore_adf"])
            >>> parser = DSJsonFormatParser(workspace)
            >>> json_str = """
            ... {
            ...     "_label_cost": -1.0,
            ...     "_label_probability": 0.5,
            ...     "_label_Action": 2,
            ...     "_labelIndex": 1,
            ...     "a": [2, 1],
            ...     "c": {
            ...         "shared": { "f": "1" },
            ...         "_multi": [{ "action": { "f": "1" } }, { "action": { "f": "2" } }]
            ...     },
            ...     "p": [0.5, 0.5]
            ... }
            ... """
            >>> example = parser.parse_json(json_str)

        Args:
            text (str): JSON string of input

        Returns:
            typing.List[Example]: List of parsed examples
        '''
        return [
            Example(_existing_example=ex, _label_type=self._workspace.label_type)
            for ex in _core._parse_line_dsjson(self._workspace._workspace, text)
        ]


DSJsonFormatReaderT = typing.TypeVar("DSJsonFormatReaderT", bound="DSJsonFormatReader")


# takes a file and uses a context manager to generate based on the contents of the file
class DSJsonFormatReader:
    def __init__(self, workspace: Workspace, file: typing.TextIO):
        """Read VW DSJson format examples from the given text file. This reader always produces lists of examples.

        Examples:
            >>> from vowpal_wabbit_next import Workspace, DSJsonFormatReader
            >>> workspace = Workspace()
            >>> with open("data.txt", "r") as f:
            ...     with DSJsonFormatReader(workspace, f) as reader:
            ...         for example in reader:
            ...               workspace.predict_one(example)


        Args:
            workspace (Workspace): Workspace object used to configure this reader
            file (typing.BinaryIO): File to read from
        """
        self._parser = DSJsonFormatParser(workspace)
        self._workspace = workspace
        self._file = file
        if not self._workspace.multiline:
            raise ValueError("Must use a multiline Workspace for dsjson format")

    def __enter__(self: DSJsonFormatReaderT) -> DSJsonFormatReaderT:
        return self

    def __exit__(
        self,
        exc_type: typing.Optional[typing.Type[BaseException]],
        exc_value: typing.Optional[BaseException],
        traceback: typing.Optional[TracebackType],
    ) -> None:
        self._file.close()

    def __iter__(self) -> typing.Iterator[typing.List[Example]]:
        if self._workspace.multiline:
            for line in self._file:
                yield self._parser.parse_json(line.rstrip())
