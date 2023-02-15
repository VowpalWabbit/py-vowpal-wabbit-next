import vowpal_wabbit_next as vw
import pytest


def test_get_metric() -> None:
    text_input = """{"_label_cost":-0.0,"_label_probability":0.05000000074505806,"_label_Action":4,"_labelIndex":3,"o":[{"v":0.0,"EventId":"13118d9b4c114f8485d9dec417e3aefe","ActionTaken":false}],"Timestamp":"2021-02-04T16:31:29.2460000Z","Version":"1","EventId":"13118d9b4c114f8485d9dec417e3aefe","a":[4,2,1,3],"c":{"FromUrl":[{"timeofday":"Afternoon","weather":"Sunny","name":"Cathy"}],"_multi":[{"_tag":"Cappucino","i":{"constant":1,"id":"Cappucino"},"j":[{"type":"hot","origin":"kenya","organic":"yes","roast":"dark"}]},{"_tag":"Cold brew","i":{"constant":1,"id":"Cold brew"},"j":[{"type":"cold","origin":"brazil","organic":"yes","roast":"light"}]},{"_tag":"Iced mocha","i":{"constant":1,"id":"Iced mocha"},"j":[{"type":"cold","origin":"ethiopia","organic":"no","roast":"light"}]},{"_tag":"Latte","i":{"constant":1,"id":"Latte"},"j":[{"type":"hot","origin":"brazil","organic":"no","roast":"dark"}]}]},"p":[0.05,0.05,0.05,0.85],"VWState":{"m":"ff0744c1aa494e1ab39ba0c78d048146/550c12cbd3aa47f09fbed3387fb9c6ec"},"_original_label_cost":-0.0}"""
    workspace = vw.Workspace(["--cb_explore_adf"], record_metrics=True)
    parser = vw.DSJsonFormatParser(workspace)
    workspace.learn_one(parser.parse_json(text_input))
    metrics = workspace.metrics
    assert len(metrics) > 0
    assert "sfm_count_learn_example_with_shared" in metrics


def test_get_metric_not_enabled() -> None:
    workspace = vw.Workspace(["--cb_explore_adf"])

    with pytest.raises(RuntimeError):
        workspace.metrics
