import typing

from vowpal_wabbit_next import _core, Workspace, Example
from types import TracebackType


class TextFormatParser:
    def __init__(self, workspace: Workspace):
        self._workspace = workspace

    def parse_line(self, text: str) -> Example:
        return _core._parse_line_text(self._workspace._workspace, text)


TextFormatReaderT = typing.TypeVar("TextFormatReaderT", bound="TextFormatReader")


# takes a file and uses a context manager to generate based on the contents of the file
class TextFormatReader:
    def __init__(self, workspace: Workspace, file: typing.TextIO):
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
                if ex._is_newline():
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
