import argparse
import json
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    destination=args.output or ROOT/"runtime_data/test_results_local.json"
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps({"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "success":result.wasSuccessful(),"scope":"Synthetic API edge cases + real CPU model fixture. Stub API tests are not real-model clinical performance."},indent=2)+"\n",encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
