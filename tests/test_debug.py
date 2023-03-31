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


def calc_depth(node, depth=1):
    if len(node.children) == 0:
        return depth
    return max(calc_depth(child, depth + 1) for child in node.children)


def test_debug_node_returned():
    workspace = vw.Workspace(enable_debug_tree=True)
    parser = vw.TextFormatParser(workspace)

    workspace.learn_one(parser.parse_line("0 | price:.23 sqft:.25 age:.05 2006"))
    workspace.learn_one(
        parser.parse_line("1 2 'second_house | price:.18 sqft:.15 age:.35 1976")
    )
    workspace.learn_one(
        parser.parse_line("0 1 0.5 'third_house | price:.053 sqft:.32 age:.87 1924")
    )

    prediction, dbg_node = workspace.predict_one(
        parser.parse_line("| price:0.25 sqft:0.8 age:0.1")
    )

    assert isinstance(dbg_node, vw.DebugNode)
    assert isinstance(dbg_node.output_prediction, type(prediction))
    # count_label, scorer, gd
    assert calc_depth(dbg_node) == 3


def test_automl_epsilon_decay_return_tree():
    workspace = vw.Workspace(
        ["--cb_explore_adf", "--automl=4", "--cover=4"], enable_debug_tree=True
    )
    parser = vw.TextFormatParser(workspace)

    ex = [
        parser.parse_line("shared | s_1"),
        parser.parse_line("0:0.1:0.25 | a:0.5 b:1"),
        parser.parse_line("| a:-1 b:-0.5"),
        parser.parse_line("| a:-2 b:-1"),
    ]

    dbg_node = workspace.learn_one(ex)
    assert isinstance(dbg_node, list)
    assert len(dbg_node) == 1
    assert calc_depth(dbg_node[0]) > 5
