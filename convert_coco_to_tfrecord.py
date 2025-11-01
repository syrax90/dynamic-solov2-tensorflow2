"""
Author: Pavel Timonin
Created: 2025-10-17
Description: This script converts a COCO-format dataset (images + annotations) into TFRecord(s).

- Skips images whose files do not exist.
- Skips annotations that don't produce a valid mask.
- Writes instance masks to TFRecord as PNG-compressed bytes (one per object).

Example
-------
python convert_coco_to_tfrecord.py \
  --images_root /path/to/images \
  --annotations /path/to/instances_train.json \
  --output /path/to/out/train.tfrecord \
  --num_shards 4

Notes
-----
- BBoxes are stored in absolute pixel coordinates (xmin, ymin, xmax, ymax).
- Category IDs are stored as-is from the COCO file.
- Masks are stored as a bytes_list feature named "image/object/mask_png"; each element is a PNG for an instance binary mask.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import List, Tuple
from coco_dataset import ann_to_mask
import numpy as np
from PIL import Image

from pycocotools.coco import COCO

import tensorflow as tf


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """
    Builds a `tf.train.Feature` containing a single bytes value.

    Args:
        value (bytes): The raw bytes to store.

    Returns:
        tf.train.Feature: A feature with `bytes_list` set to `[value]`.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_list_feature(values: List[bytes]) -> tf.train.Feature:
    """
    Builds a `tf.train.Feature` containing a list of bytes values.

    Args:
        values (List[bytes]): Iterable of byte strings to store.

    Returns:
        tf.train.Feature: A feature with `bytes_list` set to `values`.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))


def _int64_feature(value: int) -> tf.train.Feature:
    """
    Builds a `tf.train.Feature` containing a single int64 value.

    Args:
        value (int): The integer to store.

    Returns:
        tf.train.Feature: A feature with `int64_list` set to `[value]`.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(values: List[int]) -> tf.train.Feature:
    """
    Builds a `tf.train.Feature` containing a list of int64 values.

    Args:
        values (List[int]): Iterable of integers to store.

    Returns:
        tf.train.Feature: A feature with `int64_list` set to `values`.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


def _float_list_feature(values: List[float]) -> tf.train.Feature:
    """
    Builds a `tf.train.Feature` containing a list of float values.

    Args:
        values (List[float]): Iterable of floats to store.

    Returns:
        tf.train.Feature: A feature with `float_list` set to `values`.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))


def encode_mask_png(mask: np.ndarray) -> bytes:
    """
    Encodes a single-channel binary mask as PNG bytes.

    If `mask` is not `uint8`, it is cast. Masks with values in {0, 1}
    are scaled to {0, 255} for readability when decoded later.

    Args:
        mask (np.ndarray): 2D array representing the mask. Expected values are
            {0, 1} or {0, 255}. Other dtypes are cast to `np.uint8`.

    Returns:
        bytes: PNG-encoded bytes of the mask image.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # Store as 0 or 255 for better readability if decoded later
    if mask.max() <= 1:
        mask = mask * 255
    img = Image.fromarray(mask, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def coco_bbox_to_xyxy(bbox_xywh: List[float]) -> Tuple[float, float, float, float]:
    """
    Converts a COCO-format box [x, y, w, h] to [xmin, ymin, xmax, ymax].

    Args:
        bbox_xywh (List[float]): Bounding box in COCO format (top-left x/y, width, height).

    Returns:
        Tuple[float, float, float, float]: `(xmin, ymin, xmax, ymax)` as floats.
    """
    x, y, w, h = bbox_xywh
    return float(x), float(y), float(x + w), float(y + h)


def load_image_bytes_and_size(path: Path) -> Tuple[bytes, int, int, str]:
    """
    Reads an image file as bytes and returns its size and format.

    Uses PIL to determine width/height/format without decoding twice.

    Args:
        path (Path): Path to the image file.

    Returns:
        Tuple[bytes, int, int, str]: A tuple `(data, height, width, fmt)` where:
            - `data` (bytes): Raw file bytes.
            - `height` (int): Image height in pixels.
            - `width` (int): Image width in pixels.
            - `fmt` (str): Lowercased image format (e.g., `"jpeg"`, `"png"`), or
              empty string if unknown.
    """
    with open(path, "rb") as f:
        data = f.read()
    # Use PIL to get size without decoding twice
    with Image.open(io.BytesIO(data)) as img:
        width, height = img.size
        fmt = (img.format or "").lower()
    return data, height, width, fmt


def open_sharded_writers(output_pattern: str, num_shards: int):
    """
    Creates TFRecord writers for (optionally) sharded output.

    If `num_shards == 1`, a single writer is created for `output_pattern`.
    If `num_shards > 1` and `"{shard}"` is not in `output_pattern`, a suffix
    `-{shard:05d}-of-{num_shards:05d}` plus the file suffix is appended.

    Args:
        output_pattern (str): Output path or pattern. May include `"{shard}"`.
        num_shards (int): Number of shards to create (>= 1).

    Returns:
        Tuple[List[tf.io.TFRecordWriter], Callable[[int], int]]:
            - List of writers (length == `num_shards`).
            - A formatter function mapping an item index to a writer index (0..num_shards-1).
    """
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")

    # If no shard placeholder present and num_shards>1, append one.
    if num_shards == 1:
        writers = [tf.io.TFRecordWriter(output_pattern)]
        return writers, (lambda _: 0)

    if "{shard}" not in output_pattern:
        base = Path(output_pattern)
        stem = base.stem
        suffix = base.suffix or ".tfrecord"
        output_pattern = str(base.with_name(f"{stem}-{{shard:05d}}-of-{num_shards:05d}{suffix}"))

    writers = [tf.io.TFRecordWriter(output_pattern.format(shard=i)) for i in range(num_shards)]
    return writers, (lambda idx: idx % num_shards)


def build_example(
    img_bytes: bytes,
    height: int,
    width: int,
    filename: str,
    image_id: int,
    img_fmt: str,
    anns: List[dict],
) -> tf.train.Example:
    """
    Builds a `tf.train.Example` for an image and its instance annotations.

    The example includes encoded image bytes and per-instance fields such as
    bounding boxes, areas, category IDs, iscrowd flags, and PNG-encoded masks.

    Args:
        img_bytes (bytes): Raw encoded image bytes.
        height (int): Image height in pixels.
        width (int): Image width in pixels.
        filename (str): Basename of the image file.
        image_id (int): Unique image identifier (e.g., COCO image id).
        img_fmt (str): Lowercased image format (e.g., `"jpeg"`, `"png"`), may be empty.
        anns (List[dict]): COCO-style annotations for the image. Each annotation must
            contain at least `bbox`, `category_id`, and a valid `segmentation`.

    Returns:
        tf.train.Example: A serialized example containing image-level and object-level features.

    Raises:
        KeyError: If required annotation keys are missing.
        ValueError: If any annotation cannot be converted to a mask.
    """
    # Prepare per-object fields
    xmins: List[float] = []
    ymins: List[float] = []
    xmaxs: List[float] = []
    ymaxs: List[float] = []
    areas: List[float] = []
    cat_ids: List[int] = []
    iscrowds: List[int] = []
    mask_pngs: List[bytes] = []

    for ann in anns:
        # bbox
        xmin, ymin, xmax, ymax = coco_bbox_to_xyxy(ann["bbox"])  # COCO bbox: [x,y,w,h]
        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
        areas.append(float(ann.get("area", (xmax - xmin) * (ymax - ymin))))
        cat_ids.append(int(ann["category_id"]))
        iscrowds.append(int(ann.get("iscrowd", 0)))

        # mask -> PNG bytes
        mask = ann_to_mask(ann, height, width)
        mask_pngs.append(encode_mask_png(mask))

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/encoded": _bytes_feature(img_bytes),
                "image/height": _int64_feature(int(height)),
                "image/width": _int64_feature(int(width)),
                "image/filename": _bytes_feature(filename.encode("utf-8")),
                "image/id": _int64_feature(int(image_id)),
                "image/format": _bytes_feature((img_fmt or "").encode("utf-8")),
                # Object fields
                "image/object/bbox/xmin": _float_list_feature(xmins),
                "image/object/bbox/ymin": _float_list_feature(ymins),
                "image/object/bbox/xmax": _float_list_feature(xmaxs),
                "image/object/bbox/ymax": _float_list_feature(ymaxs),
                "image/object/area": _float_list_feature(areas),
                "image/object/category_id": _int64_list_feature(cat_ids),
                "image/object/iscrowd": _int64_list_feature(iscrowds),
                "image/object/mask_png": _bytes_list_feature(mask_pngs),
            }
        )
    )
    return example


def convert(
    images_root: Path,
    annotations_json: Path,
    output_pattern: str,
    num_shards: int = 1,
    allow_empty_masks: bool = False,
) -> None:
    """
    Converts COCO annotations + images into TFRecord(s) with masks.

    Iterates over images listed in `annotations_json`, filters annotations to those
    with decodable (optionally non-empty) masks, builds `tf.train.Example`s, and writes
    them to one or more TFRecord shards.

    Args:
        images_root (Path): Directory containing the image files referenced by COCO.
        annotations_json (Path): Path to the COCO annotations JSON file.
        output_pattern (str): Output TFRecord path or pattern. If `num_shards > 1`,
            include `"{shard}"` or a `-{shard}-of-N` suffix will be appended automatically.
        num_shards (int, optional): Number of shards to write. Defaults to `1`.
        allow_empty_masks (bool, optional): If `True`, keep annotations whose decoded mask
            is empty. If `False`, empty masks are dropped. Defaults to `False`.

    Returns:
        None

    Logs:
        Prints periodic progress and a final JSON summary of totals and skips.
    """
    coco = COCO(str(annotations_json))

    img_ids = sorted(coco.getImgIds())
    images = coco.loadImgs(img_ids)

    skipped_missing = 0
    skipped_no_valid_anns = 0
    total = 0
    written = 0

    writers, shard_of = open_sharded_writers(output_pattern, num_shards)

    try:
        for idx, img in enumerate(images):
            total += 1
            file_name = img["file_name"]
            img_path = images_root / file_name
            if not img_path.exists():
                skipped_missing += 1
                continue

            try:
                img_bytes, h, w, fmt = load_image_bytes_and_size(img_path)
            except Exception:
                skipped_missing += 1
                continue

            ann_ids = coco.getAnnIds(imgIds=[img["id"]], iscrowd=None)
            anns = coco.loadAnns(ann_ids)

            # Filter to annotations that have a usable segmentation/mask
            valid_anns = []
            for ann in anns:
                segm = ann.get("segmentation", None)
                if segm is None:
                    continue
                if isinstance(segm, list) and len(segm) == 0:
                    continue
                if isinstance(segm, dict) and (not segm.get("counts")):
                    continue
                try:
                    mask = ann_to_mask(ann, h, w)
                except Exception:
                    continue
                if mask is None:
                    continue
                if not allow_empty_masks and mask.sum() == 0:
                    continue
                valid_anns.append(ann)

            if len(valid_anns) == 0:
                skipped_no_valid_anns += 1
                continue

            example = build_example(
                img_bytes=img_bytes,
                height=h,
                width=w,
                filename=file_name,
                image_id=int(img["id"]),
                img_fmt=fmt,
                anns=valid_anns,
            )

            shard_id = shard_of(idx)
            writers[shard_id].write(example.SerializeToString())
            written += 1

            if written % 500 == 0:
                print(f"progress: written={written} / seen={total}")
    finally:
        for w in writers:
            w.close()

    print(
        json.dumps(
            {
                "total_images": total,
                "written": written,
                "skipped_missing_files": skipped_missing,
                "skipped_no_valid_annotations": skipped_no_valid_anns,
                "output_pattern": output_pattern,
                "num_shards": num_shards,
            },
            indent=2,
        ),
        json.dumps(
        {

                "skipped_missing_files": skipped_missing,
                "skipped_no_valid_annotations": skipped_no_valid_anns,
                "output_pattern": output_pattern,
                "num_shards": num_shards,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the COCO → TFRecord converter.

    Returns:
        argparse.Namespace: Parsed arguments with fields:
            - `images_root` (Path): Directory containing images.
            - `annotations` (Path): Path to COCO annotations JSON.
            - `output` (str): Output TFRecord path or sharded pattern.
            - `num_shards` (int): Number of shards to write.
            - `allow_empty_masks` (bool): Keep empty masks if set.
    """
    p = argparse.ArgumentParser(description="COCO -> TFRecord converter with masks")
    p.add_argument("--images_root", type=Path, required=True, help="Directory where images are stored")
    p.add_argument("--annotations", type=Path, required=True, help="Path to COCO annotations JSON")
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Output TFRecord path. If --num_shards>1, include '{shard}' placeholder or a '-{shard}-of-N' suffix will be appended."
        ),
    )
    p.add_argument("--num_shards", type=int, default=1, help="Number of shards to write")
    p.add_argument(
        "--allow_empty_masks",
        action="store_true",
        help="If set, keep anns whose decoded mask is empty (default: drop)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        images_root=args.images_root,
        annotations_json=args.annotations,
        output_pattern=args.output,
        num_shards=args.num_shards,
        allow_empty_masks=args.allow_empty_masks,
    )
