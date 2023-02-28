import vowpal_wabbit_next as vw
import pytest
import numpy as np


def test_equivalent_models() -> None:
    model = vw.Workspace(["--invert_hash=unused"])
    parser = vw.TextFormatParser(model)

    model.learn_one(parser.parse_line("1 | a b c"))

    model_after_1_learn = vw.Workspace(model_data=model.serialize())

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


def test_delta_aggregate_check_single():
    model_a = vw.Workspace([])
    text_parser = vw.TextFormatParser(model_a)

    line = "1 | a:1.1 b:0.3 c:-0.4"
    model_a.learn_one(text_parser.parse_line(line))
    weight_a = model_a.weights()[model_a.get_index_for_scalar_feature("a")][0][0]
    adaptive_a = model_a.weights()[model_a.get_index_for_scalar_feature("a")][0][1]

    model_b = vw.Workspace([])

    line = "1 | a:-1.7 b:0.9 c:1.3"
    model_b.learn_one(text_parser.parse_line(line))
    weight_b = model_b.weights()[model_b.get_index_for_scalar_feature("a")][0][0]
    adaptive_b = model_b.weights()[model_b.get_index_for_scalar_feature("a")][0][1]

    delta = vw.calculate_delta(vw.Workspace([]), model_a)
    another_delta = vw.calculate_delta(vw.Workspace([]), model_b)
    sum_deltas = vw.merge_deltas([delta, another_delta])
    new_model = vw.apply_delta(vw.Workspace([]), sum_deltas)

    ab_weight = new_model.weights()[new_model.get_index_for_scalar_feature("a")][0][0]
    ab_adaptive = new_model.weights()[new_model.get_index_for_scalar_feature("a")][0][1]

    adaptive_total = adaptive_a + adaptive_b
    assert ab_adaptive == pytest.approx(
        adaptive_a * adaptive_a / adaptive_total
        + adaptive_b * adaptive_b / adaptive_total
    )
    assert ab_weight == pytest.approx(
        weight_a * adaptive_a / adaptive_total + weight_b * adaptive_b / adaptive_total
    )


def test_delta_aggregate_check_all():
    model_a = vw.Workspace([])
    text_parser = vw.TextFormatParser(model_a)

    line = "1 | a:1.1 b:0.3 c:-0.4"
    model_a.learn_one(text_parser.parse_line(line))

    model_b = vw.Workspace([])

    line = "1 | a:-1.7 b:0.9 c:1.3"
    model_b.learn_one(text_parser.parse_line(line))

    delta = vw.calculate_delta(vw.Workspace([]), model_a)
    another_delta = vw.calculate_delta(vw.Workspace([]), model_b)
    sum_deltas = vw.merge_deltas([delta, another_delta])
    new_model = vw.apply_delta(vw.Workspace([]), sum_deltas)

    model_a_weights = model_a.weights()[:, :, 0]
    model_a_adaptives = model_a.weights()[:, :, 1]

    model_b_weights = model_b.weights()[:, :, 0]
    model_b_adaptives = model_b.weights()[:, :, 1]

    adaptive_sums = model_a_adaptives + model_b_adaptives
    # Use a masked_array to avoid divide by zero warnings for features with no data
    adaptive_sums = np.ma.masked_values(adaptive_sums, 0)

    # Ensure that corresponding indices are nonzero
    assert (
        np.ma.getmask(np.ma.masked_values(model_a_weights + model_b_weights, 0))
        == np.ma.getmask(adaptive_sums)
    ).all()

    # Bring back the zero values for masked away features
    reweighted_weights = (
        model_a_weights * model_a_adaptives / adaptive_sums
        + model_b_weights * model_b_adaptives / adaptive_sums
    ).filled(0)

    reweighted_adaptives = (
        model_a_adaptives * model_a_adaptives / adaptive_sums
        + model_b_adaptives * model_b_adaptives / adaptive_sums
    ).filled(0)

    assert np.allclose(new_model.weights()[:, :, 0], reweighted_weights)
    assert np.allclose(new_model.weights()[:, :, 1], reweighted_adaptives)
