import typing

from vowpal_wabbit_next import Example, Workspace, _core, TextFormatParser
from types import TracebackType

CacheFormatReaderT = typing.TypeVar("CacheFormatReaderT", bound="CacheFormatReader")


class CacheFormatReader:
    def __init__(self, workspace: Workspace, file: typing.BinaryIO):
        """Read VW examples in cache format from the given file.

        Examples:
            >>> from vowpal_wabbit_next import Workspace, TextFormatParser, CacheFormatWriter
            >>> workspace = Workspace()
            >>> with open("data.cache", "rb") as f:
            ...     with CacheFormatReader(workspace, f) as reader:
            ...         for example in reader:
            ...               workspace.predict_one(example)


        Args:
            workspace (Workspace): Workspace object used to configure this reader
            file (typing.BinaryIO): File to read from
        """
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
                    so_far.append(
                        Example(
                            _existing_example=next_example,
                            _label_type=self._workspace.label_type,
                        )
                    )
                next_example = self._reader._get_next()
            if len(so_far) != 0:
                yield so_far
        else:
            next_example = self._reader._get_next()
            while next_example is not None:
                yield Example(
                    _existing_example=next_example,
                    _label_type=self._workspace.label_type,
                )
                next_example = self._reader._get_next()


CacheFormatWriterT = typing.TypeVar("CacheFormatWriterT", bound="CacheFormatWriter")


class CacheFormatWriter:
    def __init__(self, workspace: Workspace, file: typing.BinaryIO):
        """Creates a VW cache file.

        Examples:
            >>> from vowpal_wabbit_next import Workspace, TextFormatParser, CacheFormatWriter
            >>> workspace = Workspace()
            >>> parser = TextFormatParser(workspace)
            >>> with open("data.cache", "wb") as f:
            ...     with CacheFormatWriter(workspace, f) as writer:
            ...         writer.write_example(parser.parse_line("1.0 | price:.18 sqft:.15 age:.35 1976"))

        Args:
            workspace (Workspace): Workspace object used to configure this writer.
            file (typing.BinaryIO): File to write cache to
        """
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
        """Write a single example to the cache file.

        Args:
            example (typing.Union[Example, typing.List[Example]]): Either a single or multiex to be written.
        """
        if isinstance(example, list):
            for e in example:
                _core._write_cache_example(
                    self._workspace._workspace, e._example, self._file
                )
            _core._write_cache_example(
                self._workspace._workspace, self._newline_example._example, self._file
            )
        else:
            _core._write_cache_example(
                self._workspace._workspace, example._example, self._file
            )
