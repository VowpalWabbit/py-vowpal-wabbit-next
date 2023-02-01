import vowpal_wabbit_next as vw


def test_learn() -> None:
    model = vw.Workspace([])
    parser = vw.TextFormatParser(model)

    pred = model.predict_one(parser.parse_line("| a"))
    assert pred == 0.0
    model.learn_one(parser.parse_line("1 | a b c"))
    pred = model.predict_one(parser.parse_line("| a"))

    # Learn should result in a non-zero prediction.
    assert pred != 0
