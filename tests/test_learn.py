import vowpal_wabbit_next as vw
import numpy as np


def test_learn() -> None:
    model = vw.Workspace()
    parser = vw.TextFormatParser(model)

    pred = model.predict_one(parser.parse_line("| a"))
    assert pred == 0.0
    model.learn_one(parser.parse_line("1 | a b c"))
    pred = model.predict_one(parser.parse_line("| a"))

    # Learn should result in a non-zero prediction.
    assert pred != 0

    idx = model.get_index_for_scalar_feature("a")

    assert model.weights()[idx][0][0] != 0


def test_predict_then_learn() -> None:
    model = vw.Workspace()
    parser = vw.TextFormatParser(model)

    pred = model.predict_then_learn_one(parser.parse_line("1 | a b c"))
    assert pred == 0.0
    pred = model.predict_one(parser.parse_line("| a"))

    # Learn should result in a non-zero prediction.
    assert pred != 0

    idx = model.get_index_for_scalar_feature("a")

    assert model.weights()[idx][0][0] != 0


def test_predict_then_learn_equivalent() -> None:
    model_learn = vw.Workspace()
    model_predict_and_learn = vw.Workspace()
    parser = vw.TextFormatParser(model_learn)

    model_learn.learn_one(parser.parse_line("1 | a b c"))
    model_learn.learn_one(parser.parse_line("2 | b d"))
    model_learn.learn_one(parser.parse_line("0.5 | b"))

    _ = model_predict_and_learn.predict_then_learn_one(parser.parse_line("1 | a b c"))
    _ = model_predict_and_learn.predict_then_learn_one(parser.parse_line("2 | b d"))
    _ = model_predict_and_learn.predict_then_learn_one(parser.parse_line("0.5 | b"))

    assert np.allclose(model_learn.weights(), model_predict_and_learn.weights())
