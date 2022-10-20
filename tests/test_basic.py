import vowpalwabbit_next as m


def test_version() -> None:
    assert m.__version__ == "0.0.1"


def test_add() -> None:
    assert m.add(1, 2) == 3


def test_sub() -> None:
    assert m.subtract(1, 2) == -1
