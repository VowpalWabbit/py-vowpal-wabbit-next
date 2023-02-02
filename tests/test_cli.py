import vowpal_wabbit_next as vw
import pytest


def test_cli_produces_output() -> None:
    driver_output, log_output = vw.run_cli_driver(["-q::", "-qab"])
    assert len(driver_output) > 0
    # the -q should produce warnings
    assert len(log_output) > 0


def test_cli_onethread() -> None:
    driver_output, _ = vw.run_cli_driver([], onethread=True)
    assert len(driver_output) > 0


def test_cli_raises_error() -> None:
    with pytest.raises(vw.CLIError) as e_info:
        _, _ = vw.run_cli_driver(["--unknown_arg"])
