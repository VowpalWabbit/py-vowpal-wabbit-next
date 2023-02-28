from io import StringIO
import vowpal_wabbit_next as vw
import logging


def test_logger_capture() -> None:
    log_stream = StringIO()
    driver_logger = logging.getLogger("vowpal_wabbit_next.driver")
    driver_logger.setLevel("INFO")
    stream_handler = logging.StreamHandler(stream=log_stream)
    driver_logger.addHandler(stream_handler)
    _ = vw.Workspace()
    driver_logger.removeHandler(stream_handler)
    assert log_stream.getvalue() != ""
