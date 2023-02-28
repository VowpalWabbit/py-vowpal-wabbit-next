import vowpal_wabbit_next as vw
import pytest


def test_simple_label() -> None:
    example = vw.Example()
    assert example.get_label() is None
    example.set_label(vw.SimpleLabel(label=1.0))
    assert isinstance(example.get_label(), vw.SimpleLabel)
    assert example.get_label().label == 1.0
    assert example.get_label().weight == 1.0
    assert example.get_label().initial == 0.0

    example.set_label(vw.MulticlassLabel(label=2))
    assert example.get_label().label == 2


def test_simple_label_still_there_after_learn() -> None:
    model = vw.Workspace()
    parser = vw.TextFormatParser(model)

    example = parser.parse_line("1 | a b c")
    assert example.get_label().label == 1
    assert isinstance(example.get_label(), vw.SimpleLabel)
    model.learn_one(example)
    assert example.get_label().label == 1
    assert isinstance(example.get_label(), vw.SimpleLabel)


def test_multiclass_label() -> None:
    example = vw.Example()
    assert example.get_label() is None
    example.set_label(vw.MulticlassLabel(label=1))
    assert isinstance(example.get_label(), vw.MulticlassLabel)
    assert example.get_label().label == 1
    assert example.get_label().weight == 1.0
    example.set_label(None)
    assert example.get_label() is None


def test_cb_label() -> None:
    example = vw.Example()
    assert example.get_label() is None
    example.set_label(vw.CBLabel(label=(3.4, 1.2), weight=0.5))
    assert isinstance(example.get_label(), vw.CBLabel)
    assert example.get_label().label == pytest.approx((0, 3.4, 1.2))

    with pytest.raises(ValueError):
        example.set_label(vw.CBLabel(label=(3.4, 1.2), shared=True))

    example.set_label(vw.CBLabel(shared=True))
    assert isinstance(example.get_label(), vw.CBLabel)
    assert example.get_label().label is None
    assert example.get_label().shared == True

    example.set_label(None)
    assert example.get_label() is None


def test_cs_label() -> None:
    example = vw.Example()
    assert example.get_label() is None
    example.set_label(vw.CSLabel(costs=[(1, 1.2)]))
    assert isinstance(example.get_label(), vw.CSLabel)
    assert len(example.get_label().costs) == 1
    assert example.get_label().costs[0] == pytest.approx((1, 1.2))

    with pytest.raises(ValueError):
        example.set_label(vw.CSLabel(costs=[(1, 1.2)], shared=True))

    example.set_label(vw.CSLabel(shared=True))
    assert isinstance(example.get_label(), vw.CSLabel)
    assert example.get_label().costs is None
    assert example.get_label().shared == True

    example.set_label(None)
    assert example.get_label() is None


def test_cs_label_parsed() -> None:
    workspace = vw.Workspace(["--csoaa_ldf=mc"])
    parser = vw.TextFormatParser(workspace)
    shared_ex = parser.parse_line("shared | a b c")
    labeled_ex = parser.parse_line("1:0.5 | a b c")
    assert isinstance(shared_ex.get_label(), vw.CSLabel)
    assert isinstance(labeled_ex.get_label(), vw.CSLabel)

    assert shared_ex.get_label().costs is None
    assert shared_ex.get_label().shared == True

    assert len(labeled_ex.get_label().costs) == 1
    assert labeled_ex.get_label().costs[0] == pytest.approx((1, 0.5))
    assert labeled_ex.get_label().shared == False
