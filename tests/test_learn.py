import vowpal_wabbit_next as vw
import numpy as np


def test_learn() -> None:
    model: vw.Workspace = vw.Workspace()
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


# https://github.com/VowpalWabbit/py-vowpal-wabbit-next/issues/80
def test_oaa_learn_crash():
    dataset = [
        "2 |text three",
        "0 |text one",
        "1 |text two",
        "5 |text six",
        "3 |text four",
        "9 |text ten",
        "7 |text eight",
        "8 |text nine",
        "6 |text seven",
    ]

    workspace = vw.Workspace(
        ["--oaa=10", "--probabilities", "--loss_function=logistic"]
    )
    parser = vw.TextFormatParser(workspace)

    for line in dataset:
        workspace.learn_one(parser.parse_line(line))


def test_igl_predict_and_learn():
    dsjson_input = """{"_label_cost": 0, "_label_probability": 0.25, "_label_Action": 1, "_labelIndex": 0, "o": [{"v": {"v=click": 1}, "_definitely_bad": false}], "a": [0, 1, 2, 3], "c": {"User": {"user=Tom": 1, "time=afternoon" : 1}, "_multi": [{"Action": {"action=politics": 1}}, {"Action": {"action=sports": 1}}, {"Action": {"action=music": 1}}, {"Action": {"action=food": 1}}]}, "p": [0.25, 0.25, 0.25, 0.25]}"""
    model = vw.Workspace(
        ["--cb_explore_adf", "--coin", "--experimental_igl", "-q", "UA"]
    )

    parser = vw.DSJsonFormatParser(model)
    examples = parser.parse_json(dsjson_input)

    assert len(examples) == 6

    pred = model.predict_one(examples)
    assert all([p == 0.25 for _, p in pred])
    model.learn_one(examples)

    assert (
        model.weights()[
            model.get_index_for_scalar_feature("user=Tom", namespace_name="User")
        ][0][0]
        != 0
    )
