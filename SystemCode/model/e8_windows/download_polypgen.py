"""Download the author's public PolypGen v3 archive; never requires credentials.

The file ID was read from the Drive folder linked by the authors' GitHub README.
No downloaded code is executed. ZIP integrity and safe member paths are checked.
"""
import argparse
import hashlib
import http.cookiejar
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

FILE_ID = "1DV-q3g35CiHHG16k13QmvuHdaDQ6dkqh"
FILENAME = "PolypGen2021_MultiCenterData_v3.zip"
EXPECTED_BYTES = 3419253506
AUTHOR_FOLDER = "https://drive.google.com/drive/folders/16uL9n84SrMt7IiQFzTUQNaJ9TbHJ8DhW"
AUTHOR_REPO = "https://github.com/DebeshJha/PolypGen"


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action, self.fields, self.active = None, {}, False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.active, self.action = True, attrs.get("action")
        if tag == "input" and self.active and attrs.get("type") == "hidden":
            self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def binary_response(opener, offset=0):
    url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
        {"id": FILE_ID, "export": "download"})
    headers = {"User-Agent": "PolypPVT-E8-research-download/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    response = opener.open(urllib.request.Request(url, headers=headers), timeout=60)
    if "text/html" in response.headers.get("Content-Type", ""):
        parser = DownloadForm()
        parser.feed(response.read(200000).decode("utf-8"))
        response.close()
        if parser.action != "https://drive.usercontent.google.com/download" or parser.fields.get("id") != FILE_ID:
            raise RuntimeError("Drive did not offer the expected public download. Login/quota/CAPTCHA is not bypassed.")
        url = parser.action + "?" + urllib.parse.urlencode(parser.fields)
        response = opener.open(urllib.request.Request(url, headers=headers), timeout=60)
    if "text/html" in response.headers.get("Content-Type", ""):
        response.close()
        raise RuntimeError("Received HTML instead of an archive; access or quota needs attention.")
    if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
        response.close()
        raise RuntimeError("Server does not support safe resume; preserve the .part file and use a new output directory.")
    return response


def safe_members(archive, destination):
    root = destination.resolve()
    for member in archive.infolist():
        name = member.filename.replace("\\", "/")
        if name.startswith("/") or ":" in name or ".." in Path(name).parts:
            raise ValueError("Unsafe ZIP path: " + name)
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError("ZIP path escaped extraction directory")
        if ((member.external_attr >> 16) & 0o170000) == 0o120000:
            raise ValueError("ZIP symlink is not allowed")
        yield member


def download(output, extract=True):
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / FILENAME
    part = output / (FILENAME + ".part")
    if not archive_path.exists():
        if shutil.disk_usage(output).free < EXPECTED_BYTES * 3:
            raise RuntimeError("Less than 3x archive size free; reserve room for extraction and checkpoints.")
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        offset = part.stat().st_size if part.exists() else 0
        started, last_print = time.monotonic(), 0.0
        with binary_response(opener, offset) as response, part.open("ab" if offset else "xb") as target:
            downloaded = offset
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_print >= 10:
                    print(f"[download] {downloaded / 2**20:.1f}/{EXPECTED_BYTES / 2**20:.1f} MiB "
                          f"({downloaded / EXPECTED_BYTES:.1%}); {(downloaded-offset)/max(1,now-started)/2**20:.1f} MiB/s", flush=True)
                    last_print = now
        if part.stat().st_size != EXPECTED_BYTES:
            raise RuntimeError("Archive byte count differs from the author's public Drive metadata; preserve partial file.")
        os.replace(part, archive_path)
    if archive_path.stat().st_size != EXPECTED_BYTES:
        raise ValueError("Existing archive has unexpected size")
    print("[verify] Computing SHA256/MD5 and ZIP CRC; this may take a few minutes.", flush=True)
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with archive_path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError("ZIP CRC failed: " + bad)
        destination = output / "extracted"
        members = list(safe_members(archive, destination))
        if extract and not (output / "extraction_complete.json").exists():
            if destination.exists():
                raise FileExistsError("Unmarked extraction directory exists; inspect it and use a new directory.")
            destination.mkdir()
            for index, member in enumerate(members, 1):
                archive.extract(member, destination)
                if index % 2000 == 0:
                    print(f"[extract] {index}/{len(members)} members", flush=True)
            (output / "extraction_complete.json").write_text(json.dumps({"sha256": sha.hexdigest(), "members": len(members)}), encoding="utf-8")
    provenance = {"dataset": "PolypGen v3", "filename": FILENAME, "bytes": archive_path.stat().st_size,
                  "source": "Author-linked public Google Drive full archive (not third-party conversion)",
                  "author_repository": AUTHOR_REPO, "author_folder": AUTHOR_FOLDER, "public_file_id": FILE_ID,
                  "canonical_entity": "https://www.synapse.org/Synapse:syn45200214",
                  "paper": "https://doi.org/10.1038/s41597-023-01981-y",
                  "license": "Archive license.txt: CC BY-NC-SA 4.0; README/paper: CC BY 4.0. Discrepancy recorded; non-commercial research only, retain attribution and share-alike restrictions for derived data; seek author clarification before redistribution/commercial use.",
                  "retrieved_utc": datetime.now(timezone.utc).isoformat(), "sha256": sha.hexdigest(), "md5": md5.hexdigest(),
                  "validation": "Expected public Drive byte count and all ZIP member CRCs passed",
                  "limitation": "SHA256/MD5 are local fingerprints, not independently published author checksums."}
    (output / "download_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("DOWNLOAD_VERIFIED", archive_path, sha.hexdigest(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "dataset/_downloads/PolypGen-v3")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()
    download(args.output.resolve(), not args.no_extract)
