import vowpal_wabbit_next as vw
import pytest


def test_save_and_load() -> None:
    model = vw.Workspace([])
    parser = vw.TextFormatParser(model)
    train_example_input = "1 | a b c"
    test_example_input = "| b c"

    train_example = parser.parse_line(train_example_input)
    model.setup_example(train_example)
    test_example = parser.parse_line(test_example_input)
    model.setup_example(test_example)

    model.learn_one(train_example)
    pred1 = model.predict_one(test_example)
    data = model.serialize()

    model2 = vw.Workspace([], model_data=data)
    parser2 = vw.TextFormatParser(model)
    test_example2 = parser2.parse_line(test_example_input)
    model2.setup_example(test_example2)
    pred2 = model2.predict_one(test_example2)

    assert pred1 == pytest.approx(pred2)
