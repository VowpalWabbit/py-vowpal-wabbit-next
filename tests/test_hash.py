import vowpal_wabbit_next as vw
import pytest


def test_get_scalar_index() -> None:
    model = vw.Workspace([])

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


@pytest.mark.skip("Not yet implemented")
def test_get_interacted_index() -> None:
    model = vw.Workspace([])

    assert 2692 == model.get_index_for_interacted_feature(
        [("a", None, " "), ("a", None, " ")]
    )
    assert 210827 == model.get_index_for_interacted_feature(
        [("thing", None, "test"), ("val", "test", "another")]
    )
