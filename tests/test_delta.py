import vowpal_wabbit_next as vw
import pytest


def test_equivalent_models() -> None:
    model = vw.Workspace([])
    parser = vw.TextFormatParser(model)
    train_example_input1 = "1 | a b c"
    train_example_input2 = "1 | b c d"
    test_example = "| a d"

    train_example = parser.parse_line(train_example_input1)
    model.setup_example(train_example)
    model.learn_one(train_example)

    base_model_data = model.serialize()
    base_model = vw.Workspace([], model_data=base_model_data)

    train_example = parser.parse_line(train_example_input2)
    model.setup_example(train_example)
    model.learn_one(train_example)

    delta = vw.calculate_delta(base_model, model)
    delta_applied_model = vw.apply_delta(base_model, delta)

    parser.parse_line(test_example)
    model.setup_example(train_example)

    assert model.predict_one(train_example) == pytest.approx(
        delta_applied_model.predict_one(train_example)
    )
