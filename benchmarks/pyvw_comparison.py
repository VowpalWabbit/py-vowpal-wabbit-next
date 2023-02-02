import pyperf
import time
import vowpalwabbit as pyvw
import vowpal_wabbit_next as vwnext

dsjson_text_input = """{"_label_cost":-0.0,"_label_probability":0.05000000074505806,"_label_Action":4,"_labelIndex":3,"o":[{"v":0.0,"EventId":"13118d9b4c114f8485d9dec417e3aefe","ActionTaken":false}],"Timestamp":"2021-02-04T16:31:29.2460000Z","Version":"1","EventId":"13118d9b4c114f8485d9dec417e3aefe","a":[4,2,1,3],"c":{"FromUrl":[{"timeofday":"Afternoon","weather":"Sunny","name":"Cathy"}],"_multi":[{"_tag":"Cappucino","i":{"constant":1,"id":"Cappucino"},"j":[{"type":"hot","origin":"kenya","organic":"yes","roast":"dark"}]},{"_tag":"Cold brew","i":{"constant":1,"id":"Cold brew"},"j":[{"type":"cold","origin":"brazil","organic":"yes","roast":"light"}]},{"_tag":"Iced mocha","i":{"constant":1,"id":"Iced mocha"},"j":[{"type":"cold","origin":"ethiopia","organic":"no","roast":"light"}]},{"_tag":"Latte","i":{"constant":1,"id":"Latte"},"j":[{"type":"hot","origin":"brazil","organic":"no","roast":"dark"}]}]},"p":[0.05,0.05,0.05,0.85],"VWState":{"m":"ff0744c1aa494e1ab39ba0c78d048146/550c12cbd3aa47f09fbed3387fb9c6ec"},"_original_label_cost":-0.0}"""


def pyvw_init_workspace():
    workspace = pyvw.Workspace("--quiet")


def vwnext_init_workspace():
    workspace = vwnext.Workspace(["--quiet"])


def pyvw_parse_dsjson():
    workspace = pyvw.Workspace("--quiet --cb_explore_adf --dsjson")
    example = workspace.parse(dsjson_text_input)
    workspace.finish_example(example)


def vwnext_parse_dsjson():
    workspace = vwnext.Workspace(["--quiet", "--cb_explore_adf"])
    parser = vwnext.DSJsonFormatParser(workspace)
    result = parser.parse_json(dsjson_text_input)


def pyvw_learn_dsjson():
    workspace = pyvw.Workspace("--quiet --cb_explore_adf --dsjson")
    workspace.learn(dsjson_text_input)


def vwnext_learn_dsjson():
    workspace = vwnext.Workspace(["--quiet", "--cb_explore_adf"])
    parser = vwnext.DSJsonFormatParser(workspace)
    workspace.learn_one(parser.parse_json(dsjson_text_input))


runner = pyperf.Runner()
runner.bench_func("pyvw_init_workspace", pyvw_init_workspace)
runner.bench_func("vwnext_init_workspace", vwnext_init_workspace)
runner.bench_func("pyvw_parse_dsjson", pyvw_parse_dsjson)
runner.bench_func("vwnext_parse_dsjson", vwnext_parse_dsjson)
runner.bench_func("pyvw_learn_dsjson", pyvw_learn_dsjson)
runner.bench_func("vwnext_learn_dsjson", vwnext_learn_dsjson)
