"""Audit repository-facing English text, local Markdown links and asset policy.

No training, network, metrics recomputation or source-provenance relaxation.
"""
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[3]
HAN=re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
TEXT={".md",".py",".json",".csv",".txt",".cmd",".sh",".ts",".tsx",".js",".jsx",".html",".yaml",".yml",".ipynb",".css",".clp"}
WEIGHT="SystemCode/model/endolens_b/weights/unet_b_expanded_seed42.pt"


def main():
    raw=subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard","-z"],cwd=ROOT)
    names=sorted(set(p.decode("utf-8") for p in raw.split(b"\0") if p))
    failures=[];texts=links=0
    for name in names:
        path=ROOT/name
        if not path.is_file():continue # staged deletion
        if HAN.search(name):failures.append("Non-English current filename:"+name)
        if path.suffix.lower() in {".pt",".pth",".ckpt"} and name!=WEIGHT:
            failures.append("Unapproved model artifact:"+name)
        if "/runtime_data/" in name or "/__pycache__/" in name:failures.append("Runtime artifact:"+name)
        if path.suffix.lower() not in TEXT:continue
        try:text=path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            failures.append("Non-UTF8 source:"+name);continue
        texts+=1
        if HAN.search(text):failures.append("Non-English project text:"+name)
        if path.suffix.lower()!=".md" or name.startswith("Miscellaneous/template_reference/") or name.endswith("UPSTREAM_POLYP_PVT.md"):
            continue # upstream/template relative links are retained historical content
        # Ignore code fences; illustrative API URLs and placeholders are not file links.
        clean=re.sub(r"(?ms)^(```|~~~).*?^\1[^\n]*$","",text)
        for match in re.finditer(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)",clean):
            target=match.group(1).strip().strip("<>")
            if urlsplit(target).scheme or target.startswith(("#","//")):continue
            target=unquote(target.split("#",1)[0].split("?",1)[0])
            if not target:continue
            links+=1
            if not (path.parent/target).exists():failures.append("Missing Markdown target:"+name+" -> "+target)
    result={"success":not failures,"text_files_checked":texts,"local_links_checked":links,"failures":failures}
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["success"] else 1)


if __name__=="__main__":main()
