import vowpal_wabbit_next as vw
import pytest


def test_save_and_load() -> None:
    model = vw.Workspace()
    parser = vw.TextFormatParser(model)
    train_example_input = "1 | a b c"
    test_example_input = "| b c"

    train_example = parser.parse_line(train_example_input)
    test_example = parser.parse_line(test_example_input)

    model.learn_one(train_example)
    pred1 = model.predict_one(test_example)
    data = model.serialize()

    model2 = vw.Workspace(model_data=data)
    parser2 = vw.TextFormatParser(model)
    test_example2 = parser2.parse_line(test_example_input)
    pred2 = model2.predict_one(test_example2)

    assert pred1 == pytest.approx(pred2)
