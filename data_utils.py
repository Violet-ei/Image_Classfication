"""Dataset helpers for class-folder image datasets."""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def safe_extract_zip(zip_path: str | os.PathLike, dst_dir: str | os.PathLike) -> Path:
    """Extract zip safely and return the dataset root.

    The expected dataset format is class-folder style:
        dataset_root/cat/xxx.jpg
        dataset_root/dog/yyy.jpg
    """
    zip_path = Path(zip_path)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in zip: {member.filename}")
        zf.extractall(dst_dir)

    # If the zip contains a single top-level directory, use it as root.
    children = [p for p in dst_dir.iterdir() if not p.name.startswith("__MACOSX")]
    dirs = [p for p in children if p.is_dir()]
    files = [p for p in children if p.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return dst_dir


def prepare_dataset_from_upload(uploaded_file) -> tuple[Path, str]:
    """Accept a Gradio uploaded file or a path. Return (dataset_root, work_dir)."""
    if uploaded_file is None:
        raise ValueError("请先上传一个数据集 zip 文件。")

    if hasattr(uploaded_file, "name"):
        zip_path = uploaded_file.name
    else:
        zip_path = str(uploaded_file)

    work_dir = tempfile.mkdtemp(prefix="imgcls_dataset_")
    dataset_root = safe_extract_zip(zip_path, work_dir)
    return dataset_root, work_dir


def discover_class_folder_dataset(dataset_root: str | os.PathLike) -> tuple[list[dict], list[str]]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    samples: list[dict] = []
    labels: list[str] = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = class_dir.name
        imgs = sorted([p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
        if not imgs:
            continue
        labels.append(label)
        for path in imgs:
            samples.append({"path": str(path), "label": label, "file": path.name})

    if not samples:
        raise ValueError(
            "没有在数据集中找到图片。请使用 class-folder 格式，例如 dataset/cat/1.jpg、dataset/dog/2.jpg。"
        )
    return samples, labels


def load_image(path: str | os.PathLike) -> Image.Image:
    return Image.open(path).convert("RGB")
