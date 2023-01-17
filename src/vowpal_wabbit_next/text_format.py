import typing

from vowpal_wabbit_next import _core, Workspace, Example
from types import TracebackType

CacheFormatReaderT = typing.TypeVar("CacheFormatReaderT", bound="CacheFormatReader")


class CacheFormatReader:
    def __init__(self, workspace: Workspace, file: typing.BinaryIO):
        self._workspace = workspace
        self._file = file
        self._reader = _core._CacheReader(self._workspace._workspace, self._file)

    def __enter__(self: CacheFormatReaderT) -> CacheFormatReaderT:
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
            next_example = self._reader._get_next()
            while next_example is not None:
                if next_example._is_newline():
                    if len(so_far) != 0:
                        yield so_far
                        so_far = []
                else:
                    so_far.append(next_example)
                next_example = self._reader._get_next()
            if len(so_far) != 0:
                yield so_far
        else:
            next_example = self._reader._get_next()
            while next_example is not None:
                yield next_example
                next_example = self._reader._get_next()


CacheFormatWriterT = typing.TypeVar("CacheFormatWriterT", bound="CacheFormatWriter")


class CacheFormatWriter:
    def __init__(self, workspace: Workspace, file: typing.BinaryIO):
        self._workspace = workspace
        self._file = file
        # TODO: workout a better way to handle this one...
        self._newline_example = TextFormatParser(workspace).parse_line("")
        _core._write_cache_header(self._workspace._workspace, self._file)

    def __enter__(self: CacheFormatWriterT) -> CacheFormatWriterT:
        return self

    def __exit__(
        self,
        exc_type: typing.Optional[typing.Type[BaseException]],
        exc_value: typing.Optional[BaseException],
        traceback: typing.Optional[TracebackType],
    ) -> None:
        self._file.close()

    def write_example(
        self, example: typing.Union[Example, typing.List[Example]]
    ) -> None:
        if isinstance(example, list):
            for e in example:
                _core._write_cache_example(self._workspace._workspace, e, self._file)
            _core._write_cache_example(
                self._workspace._workspace, self._newline_example, self._file
            )
        else:
            _core._write_cache_example(self._workspace._workspace, example, self._file)


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
