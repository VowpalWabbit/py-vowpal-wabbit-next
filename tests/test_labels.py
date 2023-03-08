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


def test_ccb_label() -> None:
    example = vw.Example()
    assert example.get_label() is None
    example.set_label(vw.CCBLabel(vw.CCBExampleType.Shared))
    assert isinstance(example.get_label(), vw.CCBLabel)
    assert example.get_label().example_type == vw.CCBExampleType.Shared

    with pytest.raises(ValueError):
        vw.CCBLabel(vw.CCBExampleType.Shared, outcome=(1.0, [(1, 0.8)]))

    with pytest.raises(ValueError):
        vw.CCBLabel(vw.CCBExampleType.Action, outcome=(1.0, [(1, 0.8)]))

    label = vw.CCBLabel(vw.CCBExampleType.Slot, outcome=(1.0, [(1, 0.8)]))
    assert label.outcome[0] == 1.0
    assert label.outcome[1][0] == pytest.approx((1, 0.8))

    label = vw.CCBLabel(vw.CCBExampleType.Slot, explicit_included_actions=[1, 2])
    assert label.explicit_included_actions == [1, 2]


def test_ccb_label_parsed() -> None:
    workspace = vw.Workspace(["--ccb_explore_adf"])
    parser = vw.TextFormatParser(workspace)
    shared_ex = parser.parse_line("ccb shared | a b c")
    assert isinstance(shared_ex.get_label(), vw.CCBLabel)
    assert shared_ex.get_label().example_type == vw.CCBExampleType.Shared
    assert shared_ex.get_label().outcome is None

    action_ex = parser.parse_line("ccb action | a b c")
    assert isinstance(action_ex.get_label(), vw.CCBLabel)
    assert action_ex.get_label().example_type == vw.CCBExampleType.Action
    assert action_ex.get_label().outcome is None

    slot_ex = parser.parse_line("ccb slot | a b c")
    assert isinstance(slot_ex.get_label(), vw.CCBLabel)
    assert slot_ex.get_label().example_type == vw.CCBExampleType.Slot
    assert slot_ex.get_label().outcome is None
    assert slot_ex.get_label().explicit_included_actions is None

    slot_ex = parser.parse_line("ccb slot 1:0.5:0.8,2:0.2 1,2,3 | a b c")
    assert slot_ex.get_label().example_type == vw.CCBExampleType.Slot
    assert slot_ex.get_label().outcome is not None
    cost, action_probs = slot_ex.get_label().outcome
    assert cost == pytest.approx(0.5)
    assert len(action_probs) == 2
    assert action_probs[0] == pytest.approx((1, 0.8))
    assert action_probs[1] == pytest.approx((2, 0.2))
    assert slot_ex.get_label().explicit_included_actions == [1, 2, 3]
