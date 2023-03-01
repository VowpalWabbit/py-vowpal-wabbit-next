import vowpal_wabbit_next as vw


def test_single() -> None:
    workspace = vw.Workspace()
    parser = vw.JsonFormatParser(workspace)
    json_str = """{"_label":-1.0,"feat1":0.5,"feat2":2}"""
    example = parser.parse_json(json_str)
    assert isinstance(example, vw.Example)
    assert isinstance(example.get_label(), vw.SimpleLabel)


def test_multi() -> None:
    workspace = vw.Workspace(["--cb_adf"])
    parser = vw.JsonFormatParser(workspace)
    json_str = """{"s_":"1","s_":"2","_labelIndex":0,"_label_Action":0,"_label_Cost":1,"_label_Probability":0.5,"_multi":[{"a_":"1","b_":"1","c_":"1"}, {"a_":"2","b_":"2","c_":"2"},{"a_":"3","b_":"3","c_":"3"}]}"""
    example = parser.parse_json(json_str)
    assert isinstance(example, list)
    assert isinstance(example[0].get_label(), vw.CBLabel)
