import typing

from vowpal_wabbit_next import _core, Workspace, Example
from types import TracebackType


class TextFormatParser:
    def __init__(self, workspace: Workspace):
        """Parse VW text format examples.

        Args:
            workspace (Workspace): Workspace object used to configure this parser
        """
        self._workspace = workspace

    def parse_line(self, text: str) -> Example:
        """Parse a single line.

        Examples:
            >>> from vowpal_wabbit_next import Workspace, TextFormatParser
            >>> workspace = Workspace()
            >>> parser = TextFormatParser(workspace)
            >>> example = parser.parse_line("1.0 | price:.18 sqft:.15 age:.35 1976")

        Args:
            text (str): Text to parse

        Returns:
            Example: Parsed example
        """
        return Example(
            _existing_example=_core._parse_line_text(self._workspace._workspace, text),
            _label_type=self._workspace.label_type,
        )


TextFormatReaderT = typing.TypeVar("TextFormatReaderT", bound="TextFormatReader")


# takes a file and uses a context manager to generate based on the contents of the file
class TextFormatReader:
    def __init__(self, workspace: Workspace, file: typing.TextIO):
        """Read VW text format examples from the given text file. This reader produces either single Examples or List[Example] based on if the given workspace is multiline or not.

        Examples:
            >>> from vowpal_wabbit_next import Workspace, TextFormatReader
            >>> workspace = Workspace()
            >>> with open("data.txt", "r") as f:
            ...     with TextFormatReader(workspace, f) as reader:
            ...         for example in reader:
            ...               workspace.predict_one(example)


        Args:
            workspace (Workspace): Workspace object used to configure this reader
            file (typing.BinaryIO): File to read from
        """
        self._parser = TextFormatParser(workspace)
        self._workspace = workspace
        self._file = file

    def __enter__(self: TextFormatReaderT) -> TextFormatReaderT:
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
            so_far: typing.List[Example] = []
            # parse until we find a newline example
            for line in self._file:
                ex = self._parser.parse_line(line)
                if ex._example._is_newline():
                    if len(so_far) != 0:
                        yield so_far
                        so_far = []
                else:
                    so_far.append(ex)
            if len(so_far) != 0:
                yield so_far
        else:
            for line in self._file:
                yield self._parser.parse_line(line)
