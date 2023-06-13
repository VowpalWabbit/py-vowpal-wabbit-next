import io
import pytest
import vowpal_wabbit_next as vw


def test_example_creation() -> None:
    example = vw.Example()
    assert example.feat_group_indices == []

    feat_group = example["a"]
    assert example.feat_group_indices == [ord("a")]

    feat_group = example["a"]

    assert len(feat_group) == 0

    feat_group.push_feature(1, 1.0)
    assert len(feat_group) == 1

    with pytest.raises(Exception) as e_info:
        feat_group.truncate_to(4)

    feat_group.truncate_to(1)
    assert len(feat_group) == 1
    feat_group.truncate_to(0)
    assert len(feat_group) == 0

    feat_group.push_feature(1, 1.0)
    assert len(feat_group) == 1

    feat_group.push_many_features([2, 3], [2.0, 3.0])
    assert len(feat_group) == 3

    feat_group.push_feature(4, 4.0)
    assert len(feat_group) == 4

    feat_group.push_many_features([5, 6], [5.0, 6.0])
    assert len(feat_group) == 6

    feat_group.truncate_to(6)
    assert len(feat_group) == 6

    feat_group.truncate_to(4)
    assert len(feat_group) == 4


def test_example_creation_index() -> None:
    example = vw.Example()
    assert example.feat_group_indices == []

    with pytest.raises(Exception) as e_info:
        feat_group = example[256]

    with pytest.raises(Exception) as e_info:
        feat_group = example["test"]

    with pytest.raises(Exception) as e_info:
        feat_group = example[-56]

    assert example.feat_group_indices == []

    assert "a" not in example
    feat_group = example["a"]
    assert len(example.feat_group_indices) == 1
    assert "a" in example

    assert 0 not in example
    feat_group = example[0]
    assert len(example.feat_group_indices) == 2
    assert 0 in example

    del example[0]
    assert 0 not in example
    assert len(example.feat_group_indices) == 1

    del example["a"]
    assert "a" not in example
    assert len(example.feat_group_indices) == 0
