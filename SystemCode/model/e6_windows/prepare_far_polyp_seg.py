#!/usr/bin/env python3
"""Prepare FAR-POLYP-SEG for the E6 independent development experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("Patient")
    return (int(suffix), path.name) if suffix.isdigit() else (10**9, path.name)


def index_files(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    result: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate stem in {folder}: {path.stem}")
        result[path.stem] = path
    return result


def prepare(source: Path, output: Path) -> None:
    if not (source / "patient_summary.csv").is_file():
        raise FileNotFoundError(f"Not a FAR-POLYP-SEG root: {source}")

    images_dir = output / "images"
    masks_dir = output / "masks"
    for folder in (images_dir, masks_dir):
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty folder: {folder}")
        folder.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    positive_patients = 0

    patient_dirs = sorted(source.glob("Patient*"), key=patient_sort_key)
    for patient_dir in patient_dirs:
        if not patient_dir.is_dir():
            continue
        polyp_files = index_files(patient_dir / "Polyp")
        mask_files = index_files(patient_dir / "BinaryMask")
        if polyp_files.keys() != mask_files.keys():
            missing_masks = sorted(polyp_files.keys() - mask_files.keys())
            missing_images = sorted(mask_files.keys() - polyp_files.keys())
            raise ValueError(
                f"Unpaired files for {patient_dir.name}; "
                f"missing masks={missing_masks}, missing images={missing_images}"
            )
        if not polyp_files:
            continue

        positive_patients += 1
        for original_stem in sorted(polyp_files):
            image_path = polyp_files[original_stem]
            mask_path = mask_files[original_stem]
            stem = f"{patient_dir.name}__{original_stem}"
            image_target = images_dir / f"{stem}{image_path.suffix.lower()}"
            mask_target = masks_dir / f"{stem}{mask_path.suffix.lower()}"
            shutil.copy2(image_path, image_target)
            shutil.copy2(mask_path, mask_target)

            metadata_rows.append(
                {
                    "stem": stem,
                    "source": "FAR-POLYP-SEG",
                    "group_id": patient_dir.name,
                }
            )
            manifest_rows.append(
                {
                    "stem": stem,
                    "group_id": patient_dir.name,
                    "source_image": image_path.relative_to(source).as_posix(),
                    "source_mask": mask_path.relative_to(source).as_posix(),
                    "image_sha256": sha256(image_target),
                    "mask_sha256": sha256(mask_target),
                }
            )

    if not metadata_rows:
        raise ValueError(f"No paired polyp images found under: {source}")

    with (output / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "source", "group_id"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    with (output / "source_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "stem",
            "group_id",
            "source_image",
            "source_mask",
            "image_sha256",
            "mask_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    shutil.copy2(source / "patient_summary.csv", output / "patient_summary.csv")
    print(f"Prepared {len(metadata_rows)} image-mask pairs from {positive_patients} patients")
    print(f"Output: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
