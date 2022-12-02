from io import StringIO
import vowpal_wabbit_next.learner
import logging


def test_logger_capture() -> None:
    log_stream = StringIO()
    driver_logger = logging.getLogger("vowpal_wabbit_next.driver")
    driver_logger.setLevel("INFO")
    stream_handler = logging.StreamHandler(stream=log_stream)
    driver_logger.addHandler(stream_handler)
    learner = vowpal_wabbit_next.learner._LearnerBase([])
    driver_logger.removeHandler(stream_handler)
    assert log_stream.getvalue() != ""
