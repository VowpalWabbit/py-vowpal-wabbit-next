import vowpal_wabbit_next as vw
import sys

print(vw.run_cli_driver(sys.argv[1:]))
