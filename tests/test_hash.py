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


@pytest.mark.skip("Not yet implemented")
def test_get_interacted_index() -> None:
    model = vw.Workspace([])

    assert 2692 == model.get_index_for_interacted_feature(
        [("a", None, " "), ("a", None, " ")]
    )
    assert 210827 == model.get_index_for_interacted_feature(
        [("thing", None, "test"), ("val", "test", "another")]
    )
