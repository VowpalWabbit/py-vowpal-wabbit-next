import os
import vowpal_wabbit_next as vw
import pytest
import pathlib


def test_cli_produces_output() -> None:
    driver_output, log_output = vw.run_cli_driver(["-q::", "-qab"])
    assert len(driver_output) > 0
    # the -q should produce warnings
    assert len(log_output) > 0


def test_cli_onethread() -> None:
    driver_output, _ = vw.run_cli_driver([], onethread=True)
    assert len(driver_output) > 0


def test_cli_cwd() -> None:
    original_cwd = pathlib.Path(os.getcwd())

    # Can't find file in current dir
    with pytest.raises(vw.CLIError) as e_info:
        _, _ = vw.run_cli_driver(["--data=rcv1_small.dat"])

    data_dir = pathlib.Path(__file__).parent.resolve() / "data"
    # Can find file in data dir
    driver_output, _ = vw.run_cli_driver(["--data=rcv1_small.dat"], cwd=data_dir)

    # ensure the cwd is restored
    assert pathlib.Path(os.getcwd()) == original_cwd
    assert len(driver_output) > 0


def test_cli_cwd_restore_on_fail() -> None:
    original_cwd = pathlib.Path(os.getcwd())
    data_dir = pathlib.Path(__file__).parent.resolve() / "data"
    with pytest.raises(vw.CLIError) as e_info:
        driver_output, _ = vw.run_cli_driver(
            ["--data=rcv1_small.dat", "--bad_opt"], cwd=data_dir
        )

    assert pathlib.Path(os.getcwd()) == original_cwd


def test_cli_raises_error() -> None:
    with pytest.raises(vw.CLIError) as e_info:
        _, _ = vw.run_cli_driver(["--unknown_arg"])
