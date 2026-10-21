"""Run isolated synthetic research regression suites, never formal experiments."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GROUPS={
    "early":["e5_windows/test_e5.py","e6_windows/test_e6.py","e7a_windows/test_e7a.py","e7a2_windows/test_e7a2.py",
             "e7b0_windows/test_e7b0.py","e7b1_windows/test_e7b1.py","e7b2_windows/test_e7b2.py"],
    "training":["e8_windows/test_e8.py","e8_windows/test_holdout_analysis.py","e8_repeat_windows/test_e8_repeat.py",
                "e9_balance_windows/test_e9_balance.py"],
    "fusion":["e10a_windows/test_e10a.py","e10b_windows/test_e10b.py","e10c_windows/test_e10c.py",
              "e10d_windows/test_e10d.py","e11_background_windows/test_e11.py"],
    "teammate":["unet_edges/test_boundary.py","polyp_outline/test_predict.py"]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--group",choices=[*GROUPS,"all"],default="all")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    scripts=[s for rows in GROUPS.values() for s in rows] if args.group=="all" else GROUPS[args.group]
    output=args.output or ROOT/"runs/publication_checks"/(args.group+".json")
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    results=[]
    for script in scripts:
        run=subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=run.stdout.decode("utf-8",errors="replace")
        logpath=ROOT/"runs/publication_checks/logs"/(Path(script).parent.name+"_"+Path(script).stem+".log")
        logpath.parent.mkdir(parents=True,exist_ok=True);logpath.write_text(log,encoding="utf-8")
        counts=re.findall(r"Ran (\d+) tests?",log)
        row={"script":script,"exit_code":run.returncode,"tests":sum(map(int,counts)) if counts else None}
        results.append(row);print(json.dumps(row),flush=True)
    summary={"group":args.group,"success":all(r["exit_code"]==0 for r in results),"suites":results,
             "tests":sum(r["tests"] or 0 for r in results),"scope":"CPU/synthetic regression, not formal training, patient evaluation or a deployment benchmark"}
    output.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    if not summary["success"]:
        for r in results:
            if r["exit_code"]:print("FAILED",r["script"],"see ignored runs/publication_checks/logs",flush=True)
    raise SystemExit(0 if summary["success"] else 1)


if __name__=="__main__":main()
