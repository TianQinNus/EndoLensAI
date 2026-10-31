"""Install the unchanged E13 checkpoint from verified Drive parts or a full file.

Uses the Python standard library only. Does not download or execute model code.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path, expected_size, expected_hash):
    if not path.is_file():
        raise FileNotFoundError(f"Missing download: {path}")
    if path.stat().st_size != expected_size or sha256(path) != expected_hash:
        raise ValueError(f"Size/SHA256 mismatch: {path.name}. Download the original file again.")


def install(parts_dir=None, source=None, check=False):
    manifest = json.loads((ROOT / "weights_download.json").read_text(encoding="utf-8"))
    model = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
    if manifest["weights_sha256"] != model["weights_sha256"]:
        raise ValueError("Download manifest and frozen model identity differ")
    destination = ROOT / model["weights_file"]
    if destination.exists():
        verify_file(destination, manifest["weights_size_bytes"], manifest["weights_sha256"])
        print("WEIGHT_VERIFIED", destination)
        return destination
    if check:
        raise FileNotFoundError(f"Weight not installed: {destination}")
    if source is not None:
        source = Path(source).expanduser().resolve()
        verify_file(source, manifest["weights_size_bytes"], manifest["weights_sha256"])
        inputs = [source]
    else:
        folder = Path(parts_dir).expanduser().resolve()
        inputs = []
        for part in manifest["parts"]:
            path = folder / part["file"]
            verify_file(path, part["size_bytes"], part["sha256"])
            inputs.append(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".e13-install-", dir=destination.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            for path in inputs:
                with path.open("rb") as stream:
                    shutil.copyfileobj(stream, output, length=1024 * 1024)
        verify_file(temporary, manifest["weights_size_bytes"], manifest["weights_sha256"])
        created = False
        try:
            with destination.open("xb") as output, temporary.open("rb") as stream:
                created = True
                shutil.copyfileobj(stream, output, length=1024 * 1024)
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            raise
    finally:
        temporary.unlink(missing_ok=True)
    verify_file(destination, manifest["weights_size_bytes"], manifest["weights_sha256"])
    print("WEIGHT_INSTALLED", destination)
    print("SHA256", manifest["weights_sha256"])
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--parts-dir", type=Path, help="Folder containing both original .part01/.part02 downloads")
    group.add_argument("--source", type=Path, help="An existing complete .pt with the released SHA256")
    group.add_argument("--check", action="store_true", help="Check an already installed weight without loading PyTorch")
    args = parser.parse_args()
    try:
        install(args.parts_dir, args.source, args.check)
    except (OSError, ValueError) as error:
        parser.exit(1, f"WEIGHT_INSTALL_FAILED: {error}\n")
