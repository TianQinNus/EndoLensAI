"""Explicitly refresh the standalone bundle manifest after reviewed changes.

Model identity/weights are immutable. This does not waive source or test checks.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/"endolens_b"


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--refresh-manifest",action="store_true",required=True)
    parser.parse_args()
    config=json.loads((ROOT/"model.json").read_text(encoding="utf-8"))
    expected="1e4531592942ac8683b3ea7ea7ac9af6e77e03b08a2e3481942cfbbb24f657bb"
    if config["weights_sha256"]!=expected or sha256(ROOT/config["weights_file"])!=expected:
        raise ValueError("Retained weight is incomplete or changed; retrieve it, do not reseal a different model")
    files={}
    for path in sorted(ROOT.rglob("*")):
        rel=path.relative_to(ROOT)
        if not path.is_file() or path.name=="MANIFEST.json" or path.suffix==".pyc":continue
        if any(p in ("__pycache__","runtime_data","single_frame_output") or p.startswith("_test_") for p in rel.parts):continue
        files[rel.as_posix()]=sha256(path)
    (ROOT/"MANIFEST.json").write_text(json.dumps({"model_version":config["model_version"],"files":files},indent=2)+"\n",encoding="utf-8")
    print("REVIEWED_BUNDLE_SEALED",len(files),"files; unchanged B weight")


if __name__=="__main__":main()
