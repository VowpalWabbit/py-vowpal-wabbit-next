import vowpal_wabbit_next as vw
import pytest


@pytest.mark.skip(
    reason="Requires this fix to enable: https://github.com/VowpalWabbit/vowpal_wabbit/pull/4483"
)
def test_equivalent_models() -> None:
    model = vw.Workspace(["--invert_hash=unused"])
    parser = vw.TextFormatParser(model)

    model.learn_one(parser.parse_line("1 | a b c"))

    model_after_1_learn = vw.Workspace([], model_data=model.serialize())

    model.learn_one(parser.parse_line("1 | d e f"))

    delta_of_second_learn = vw.calculate_delta(model_after_1_learn, model)
    model_after_1_learn_and_delta_of_second_learn_applied = vw.apply_delta(
        model_after_1_learn, delta_of_second_learn
    )

    test_example = "| d"
    assert model.predict_one(parser.parse_line(test_example)) == pytest.approx(
        model_after_1_learn_and_delta_of_second_learn_applied.predict_one(
            parser.parse_line(test_example)
        )
    )
