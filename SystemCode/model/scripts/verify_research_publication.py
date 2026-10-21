"""Verify executed imports and presentation-only English source migration.

Never rewrites an old protocol hash or makes changed code pass old preflight.
"""
import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
ARCHIVE="c00b15fae599ae00d788393f366016e4e9c9516b"
REGISTRY=ROOT/"Miscellaneous/model_artifacts/RESEARCH_CODE_IMPORT.json"
MIGRATION=ROOT/"Miscellaneous/model_artifacts/ENGLISH_SOURCE_MIGRATION.json"
HAN=re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
# Some f-string fragments contain only full-width punctuation, not Han letters.
NON_ENGLISH_TEXT=re.compile(r"[\u3000-\u303f\uff00-\uffef\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def digest(data):return hashlib.sha256(data).hexdigest()
def old_bytes(name):return subprocess.check_output(["git","show",f"{ARCHIVE}:{name}"],cwd=ROOT)


def compare(old,new,where="root"):
    if isinstance(old,ast.AST):
        if type(old) is not type(new):raise ValueError("AST node changed:"+where)
        for field,value in ast.iter_fields(old):
            other=getattr(new,field)
            if isinstance(old,ast.Constant) and field=="value" and isinstance(value,str) and NON_ENGLISH_TEXT.search(value):
                if not isinstance(other,str) or NON_ENGLISH_TEXT.search(other):raise ValueError("Not an English text replacement:"+where)
                continue
            compare(value,other,where+"."+field)
    elif isinstance(old,list):
        if not isinstance(new,list) or len(old)!=len(new):raise ValueError("AST list changed:"+where)
        for i,(a,b) in enumerate(zip(old,new)):compare(a,b,where+f"[{i}]")
    elif old!=new:raise ValueError("Non-presentation AST value changed:"+where)


def collect():
    registry=json.loads(REGISTRY.read_text(encoding="utf-8"))
    records=[];changed=[]
    for item in registry["files"]:
        name=item["path"];original=old_bytes(name)
        if digest(original)!=item["sha256"]:raise ValueError("Archive import differs:"+name)
        current=(ROOT/name).read_bytes()
        if digest(current)!=item["sha256"]:
            if not name.endswith(".py"):raise ValueError("Frozen non-Python artifact changed:"+name)
            compare(ast.parse(original.decode("utf-8-sig")),ast.parse(current.decode("utf-8-sig")),name)
            changed.append({"path":name,"executed_sha256":digest(original),"english_sha256":digest(current),
                            "change":"Only originally non-English report text constants; all other AST fields unchanged"})
        records.append(name)
    name="SystemCode/model/polyp_outline/predict.py"
    before=old_bytes(name);after=(ROOT/name).read_bytes()
    compare(ast.parse(before.decode("utf-8-sig")),ast.parse(after.decode("utf-8-sig")),name)
    if before!=after:changed.append({"path":name,"executed_sha256":digest(before),"english_sha256":digest(after),
                                    "change":"English CLI description only; teammate inference computation unchanged"})
    return {"date":"2026-10-22","original_source_commit":ARCHIVE,"original_imported_files_verified":len(records),
            "changes":changed,"metric_values_or_gates_changed":False,"old_preflight_checks_relaxed":False,
            "recovery":"Use original archived commit and complete matching private artifacts for historical recovery. English sources must not silently resume old source-hash-locked runs."}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--record",action="store_true")
    args=parser.parse_args();result=collect()
    if args.record:
        if MIGRATION.exists():raise FileExistsError("Migration already recorded; verify instead of overwriting")
        MIGRATION.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    elif result!=json.loads(MIGRATION.read_text(encoding="utf-8")):
        raise ValueError("English source changed after reviewed migration")
    print("PASS_RESEARCH_PUBLICATION",len(result["changes"]),"presentation-only files; original archive verified")


if __name__=="__main__":main()
