from typing import Generator, List, Optional, Tuple
from vowpal_wabbit_next import _core
import contextlib
from pathlib import Path
import os


# From: https://stackoverflow.com/questions/41742317/how-can-i-change-directory-with-python-pathlib
@contextlib.contextmanager
def _working_directory(path: Path) -> Generator[None, None, None]:
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


class CLIError(Exception):
    def __init__(self, message: str, driver_output: str, log_output: List[str]):
        """Represents a failure when running the CLI. Exposes output for additional context.

        Args:
            message (str): Error message
            driver_output (str): Output of the VW driver
            log_output (List[str]): List of logs messages
        """
        super().__init__(message)
        self.driver_output = driver_output
        self.log_output = log_output


def run_cli_driver(
    args: List[str], *, onethread: bool = False, cwd: Optional[Path] = None
) -> Tuple[str, List[str]]:
    """Is the equivalent of running the VW command line tool with the given command line.

    There are a few differences:

    * Any input from stdin is not supported
    * The argfile input to command line is not supported
    * If any place in VW writes to stdout, stderr directly it is not captured. This means that `--version` and `--help` are not currently captured.

    .. warning::
        This is an experimental feature.

    Examples:
        >>> from vowpal_wabbit_next import run_cli_driver
        >>> driver_output, logs = run_cli_driver(["-d", "my_data.txt"])

        You can use `shlex` to split a command line:

        >>> from vowpal_wabbit_next import run_cli_driver
        >>> import shlex
        >>> driver_output, logs = run_cli_driver(shlex.split("-d my_data.txt"))

    Args:
        args (List[str]): Arguments to be passed to the command line driver
        onethread (bool, optional): Whether to use background thread for parsing. If False, a background thread is used for parsing. If True, no background threads are used and everything is done in the foreground of this call.
        cwd (Optional[Path], optional): The current working directory to use for the command line driver. If None, the current working directory is used.

    Raises:
        CLIError: If there is any error raised by execution.

    Returns:
        Tuple[str, List[str]]: driver output and log messages respectively as a tuple
    """
    with _working_directory(cwd) if cwd is not None else contextlib.nullcontext():
        error_info, driver_output, log_output = _core._run_cli_driver(
            args, onethread=onethread
        )
        if error_info is not None:
            raise CLIError(error_info, driver_output, log_output)

        return (driver_output, log_output)
