import vowpal_wabbit_next as vw


def test_get_scalar_index() -> None:
    model = vw.Workspace()

    assert 69985 == model.get_index_for_scalar_feature(
        "val", feature_value="test", namespace_name="another"
    )
    assert 92594 == model.get_index_for_scalar_feature("a")
    assert 148099 == model.get_index_for_scalar_feature("thing", namespace_name="test")
    assert 163331 == model.get_index_for_scalar_feature("b")

    model_with_wpp = vw.Workspace(["--automl=4", "--cb_adf"])
    # This hash causes a truncation since the wpp multiplies beyond the end of the weight array
    # We are ensuring the wrap around is calculated correctly.
    assert 45701 == model_with_wpp.get_index_for_scalar_feature(
        "feature", namespace_name="namespace"
    )


def test_get_weight_interleaved_models() -> None:
    model = vw.Workspace(["--automl=4", "--cb_adf"])
    parser = vw.TextFormatParser(model)

    example = [
        parser.parse_line("shared | s_1"),
        parser.parse_line("0:1.5:0.25 |namespace feature a:0.5 b:1"),
        parser.parse_line("| a:-1 b:-0.5"),
        parser.parse_line("| a:-2 b:-1"),
    ]
    model.learn_one(example)

    assert (
        model.weights()[
            model.get_index_for_scalar_feature("feature", namespace_name="namespace")
        ][0][0]
        != 0
    )
    assert (
        model.weights()[
            model.get_index_for_scalar_feature("feature", namespace_name="namespace")
        ][1][0]
        != 0
    )
    assert (
        model.weights()[
            model.get_index_for_scalar_feature("feature", namespace_name="namespace")
        ][2][0]
        != 0
    )
    assert (
        model.weights()[
            model.get_index_for_scalar_feature("feature", namespace_name="namespace")
        ][3][0]
        != 0
    )
    assert (
        model.weights()[model.get_index_for_scalar_feature("unknown_feature")][0][0]
        == 0
    )
    assert (
        model.weights()[model.get_index_for_scalar_feature("unknown_feature")][1][0]
        == 0
    )
    assert (
        model.weights()[model.get_index_for_scalar_feature("unknown_feature")][2][0]
        == 0
    )
    assert (
        model.weights()[model.get_index_for_scalar_feature("unknown_feature")][3][0]
        == 0
    )


# TODO: uncomment when implemented
# @pytest.mark.skip("Not yet implemented")
# def test_get_interacted_index() -> None:
#     model = vw.Workspace()

#     assert 2692 == model.get_index_for_interacted_feature(
#         [("a", None, " "), ("a", None, " ")]
#     )
#     assert 210827 == model.get_index_for_interacted_feature(
#         [("thing", None, "test"), ("val", "test", "another")]
#     )
