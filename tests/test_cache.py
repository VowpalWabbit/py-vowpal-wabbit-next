import io
import vowpal_wabbit_next as vw
from textwrap import dedent


def test_write_and_read_cache() -> None:
    text_input = io.StringIO(
        dedent(
            """1 | a b c
    0 | d e f
    """
        )
    )

    cache_buffer = io.BytesIO()
    read_buffer = io.BytesIO()
    workspace = vw.Workspace()
    write_counter = 0
    with vw.TextFormatReader(workspace, text_input) as reader:
        with vw.CacheFormatWriter(workspace, cache_buffer) as writer:
            for example in reader:
                writer.write_example(example)
                write_counter += 1

            read_buffer = io.BytesIO(cache_buffer.getvalue())

    read_counter = 0
    with vw.CacheFormatReader(workspace, read_buffer) as reader:
        for example in reader:
            assert isinstance(example, vw.Example)
            read_counter += 1

    assert write_counter == read_counter
