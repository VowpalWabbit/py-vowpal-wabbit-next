from io import StringIO
import vowpalwabbit_next.learner
import logging


def test_logger_capture() -> None:
    log_stream = StringIO()
    driver_logger = logging.getLogger("vowpalwabbit_next.driver")
    driver_logger.setLevel("INFO")
    stream_handler = logging.StreamHandler(stream=log_stream)
    driver_logger.addHandler(stream_handler)
    learner = vowpalwabbit_next.learner._LearnerBase([])
    driver_logger.removeHandler(stream_handler)
    assert log_stream.getvalue() != ""
