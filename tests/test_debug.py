import json
import vowpal_wabbit_next as vw
import pytest


def test_json_weights() -> None:
    model = vw.Workspace(["--noconstant"], record_feature_names=True)
    parser = vw.TextFormatParser(model)

    model.learn_one(parser.parse_line("1 | a b c"))

    json_weights = json.loads(model.json_weights(include_feature_names=True))
    assert len(json_weights["weights"]) == 3
    all_names = set()
    for weight in json_weights["weights"]:
        assert len(weight["terms"]) == 1
        all_names.add(weight["terms"][0]["name"])

    assert all_names == {"a", "b", "c"}


def test_json_weights_feat_name_without_constructor_enabled() -> None:
    model = vw.Workspace()
    with pytest.raises(RuntimeError):
        model.json_weights(include_feature_names=True)


def test_readable_model():
    model = vw.Workspace(record_feature_names=True)
    parser = vw.TextFormatParser(model)
    model.learn_one(parser.parse_line("1 | MY_SUPER_UNIQUE_FEATURE_NAME"))
    assert "MY_SUPER_UNIQUE_FEATURE_NAME" in model.readable_model(
        include_feature_names=True
    )


def test_readable_model_feat_name_without_constructor_enabled():
    model = vw.Workspace()
    with pytest.raises(RuntimeError):
        model.readable_model(include_feature_names=True)
