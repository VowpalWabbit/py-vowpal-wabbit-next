import typing

from vowpal_wabbit_next import _core, Workspace, Example
from types import TracebackType


class DSJsonFormatParser:
    def __init__(self, workspace: Workspace):
        self._workspace = workspace

    def parse_line(self, text: str) -> typing.List[Example]:
        return _core._parse_line_dsjson(self._workspace._workspace, text)


DSJsonFormatReaderT = typing.TypeVar("DSJsonFormatReaderT", bound="DSJsonFormatReader")


# takes a file and uses a context manager to generate based on the contents of the file
class DSJsonFormatReader:
    def __init__(self, workspace: Workspace, file: typing.TextIO):
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
                print(line.rstrip())
                yield self._parser.parse_line(line.rstrip())
