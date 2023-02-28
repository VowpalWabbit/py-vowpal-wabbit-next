import io
import vowpal_wabbit_next as vw
from textwrap import dedent


def test_simple_input() -> None:
    text_input = io.StringIO(
        dedent(
            """1 | a b c
        0 | d e f
    """
        )
    )
    workspace = vw.Workspace()
    counter = 0
    with vw.TextFormatReader(workspace, text_input) as reader:
        for example in reader:
            # perform assert example is not a list
            counter += 1
            assert isinstance(example, vw.Example)

    assert counter == 2


def test_cb_input() -> None:
    text_input = io.StringIO(
        dedent(
            """shared | s_1
        0:1.5:0.25 | a:0.5 b:1
        | a:-1 b:-0.5
        | a:-2 b:-1

        shared | s_1
        0:1.5:0.25 | a:0.5 b:1
        | a:-1 b:-0.5
        | a:-2 b:-1

        shared | s_1
        0:-1.5:0.5 | a:0.5 b:1
        | a:-1 b:-0.5
        | a:2 b:-1
    """
        )
    )
    workspace = vw.Workspace(["--cb_explore_adf"])
    counter = 0
    with vw.TextFormatReader(workspace, text_input) as reader:
        for example in reader:
            # perform assert example is not a list
            counter += 1
            assert isinstance(example, list)

    assert counter == 3
