# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0

"""Frame table ``frames/<run>.parquet``: columns, storage, and metadata.

Implements data/schema.md §3.1 (columns), §3.2 (storage), and §3.3 (file
metadata).
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

SCHEMA_VERSION = "2.0.0"
METADATA_KEY = "duckiebot"

LED_COLUMNS: list[str] = [f"led_{k:02d}" for k in range(25)]
COUNT_COLUMNS = ["marker_n", "hub_n"]
FLOAT32_COLUMNS = ["marker_x", "marker_y", "hub_x", "hub_y", "led_x", "led_y"]
FLOAT16_COLUMNS = [
    "marker_cxx",
    "marker_cyy",
    "marker_cxy",
    "hub_cxx",
    "hub_cyy",
    "hub_cxy",
]
# Columns that may be omitted on writing; they are then stored as null (§3.1:
# a run without a sync LED has null LED columns).
OPTIONAL_COLUMNS = frozenset([*LED_COLUMNS, "led_x", "led_y"])
# Spoke tracker of the floor runs (tracking.spokes): given all together or not
# at all, and stored after the columns of FRAME_SCHEMA, so frame tables tracked
# without them stay valid.
SPOKE_COLUMNS: list[str] = ["spoke_angle", "spoke_angle_sd", "spoke_contrast"]

FRAME_SCHEMA = pa.schema(
    [
        pa.field("frame", pa.uint32(), nullable=False),
        pa.field("pts", pa.int64(), nullable=False),
        pa.field("marker_n", pa.uint32(), nullable=False),
        pa.field("marker_x", pa.float32()),
        pa.field("marker_y", pa.float32()),
        pa.field("marker_cxx", pa.float16()),
        pa.field("marker_cyy", pa.float16()),
        pa.field("marker_cxy", pa.float16()),
        pa.field("hub_n", pa.uint32(), nullable=False),
        pa.field("hub_x", pa.float32()),
        pa.field("hub_y", pa.float32()),
        pa.field("hub_cxx", pa.float16()),
        pa.field("hub_cyy", pa.float16()),
        pa.field("hub_cxy", pa.float16()),
        pa.field("led_x", pa.float32()),
        pa.field("led_y", pa.float32()),
        *[pa.field(name, pa.uint8()) for name in LED_COLUMNS],
    ]
)

REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "schema_version",
    "run",
    "video_file",
    "video_sha256",
    "video_bytes",
    "width",
    "height",
    "rotation",
    "time_base",
    "fps_nominal",
    "camera",
    "mode",
    "marker_colour",
    "hub_colour",
    "marker_region",
    "hub_region",
    "led_grid",
    "decoder",
    "tracker",
    "tracked",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FRACTION = re.compile(r"^[1-9]\d*/[1-9]\d*$")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Check the §3.3 metadata keys and the values the code relies on.

    Args:
        metadata: The metadata dictionary.

    Raises:
        ValueError: Naming the first missing or invalid key.
    """
    if not isinstance(metadata, dict):
        raise ValueError(f"{METADATA_KEY} metadata must be a JSON object")
    for key in REQUIRED_METADATA_KEYS:
        if key not in metadata:
            raise ValueError(f"metadata key {key!r} is missing")
    checks: list[tuple[str, bool, str]] = [
        (
            "schema_version",
            metadata["schema_version"] == SCHEMA_VERSION,
            f"must be {SCHEMA_VERSION!r}",
        ),
        (
            "run",
            isinstance(metadata["run"], str) and bool(metadata["run"]),
            "must be a non-empty string",
        ),
        ("video_file", isinstance(metadata["video_file"], str), "must be a string"),
        (
            "video_sha256",
            isinstance(metadata["video_sha256"], str)
            and bool(_SHA256.match(metadata["video_sha256"])),
            "must be 64 lower-case hex digits",
        ),
        (
            "video_bytes",
            _is_int(metadata["video_bytes"]) and metadata["video_bytes"] >= 0,
            "must be a non-negative integer",
        ),
        (
            "width",
            _is_int(metadata["width"]) and metadata["width"] > 0,
            "must be a positive integer",
        ),
        (
            "height",
            _is_int(metadata["height"]) and metadata["height"] > 0,
            "must be a positive integer",
        ),
        ("rotation", _is_number(metadata["rotation"]), "must be a number"),
        (
            "time_base",
            isinstance(metadata["time_base"], str)
            and bool(_FRACTION.match(metadata["time_base"])),
            "must be a fraction such as '1/90000'",
        ),
        ("fps_nominal", _is_number(metadata["fps_nominal"]), "must be a number"),
        ("mode", metadata["mode"] in ("stand", "floor"), "must be 'stand' or 'floor'"),
        (
            "marker_colour",
            isinstance(metadata["marker_colour"], str),
            "must be a string",
        ),
        ("hub_colour", isinstance(metadata["hub_colour"], str), "must be a string"),
        ("tracked", isinstance(metadata["tracked"], str), "must be an ISO 8601 string"),
    ]
    for key, ok, rule in checks:
        if not ok:
            raise ValueError(f"metadata key {key!r} {rule}, got {metadata[key]!r}")


def time_base_seconds(metadata: dict[str, Any]) -> Fraction:
    """Return the seconds per ``pts`` unit as an exact fraction."""
    return Fraction(metadata["time_base"])


def _float_array(values: np.ndarray, dtype: type, pa_type: pa.DataType) -> pa.Array:
    a = np.asarray(values, dtype=dtype)
    return pa.array(a, type=pa_type, mask=np.isnan(a))


def _build_table(columns: dict[str, np.ndarray]) -> pa.Table:
    spokes = [name for name in SPOKE_COLUMNS if name in columns]
    if spokes and len(spokes) != len(SPOKE_COLUMNS):
        raise ValueError(f"spoke columns must all be given: {', '.join(SPOKE_COLUMNS)}")
    unknown = sorted(set(columns) - set(FRAME_SCHEMA.names) - set(SPOKE_COLUMNS))
    if unknown:
        raise ValueError(f"unknown frame table columns: {', '.join(unknown)}")
    missing = [
        n for n in FRAME_SCHEMA.names if n not in columns and n not in OPTIONAL_COLUMNS
    ]
    if missing:
        raise ValueError(f"missing frame table columns: {', '.join(missing)}")
    lengths = {name: len(np.asarray(v)) for name, v in columns.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"frame table columns differ in length: {lengths}")
    n = lengths["frame"]
    arrays = []
    for field in FRAME_SCHEMA:
        name = field.name
        if name not in columns:
            arrays.append(pa.nulls(n, type=field.type))
        elif name in FLOAT32_COLUMNS:
            arrays.append(_float_array(columns[name], np.float32, pa.float32()))
        elif name in FLOAT16_COLUMNS:
            arrays.append(_float_array(columns[name], np.float16, pa.float16()))
        elif name in LED_COLUMNS:
            raw = np.asarray(columns[name])
            if raw.dtype.kind == "f":
                null = np.isnan(raw)
                arrays.append(
                    pa.array(
                        np.where(null, 0, raw).astype(np.uint8),
                        type=pa.uint8(),
                        mask=null,
                    )
                )
            else:
                arrays.append(pa.array(raw.astype(np.uint8), type=pa.uint8()))
        else:
            arrays.append(
                pa.array(
                    np.asarray(columns[name]).astype(field.type.to_pandas_dtype()),
                    type=field.type,
                )
            )
    schema = FRAME_SCHEMA
    if spokes:
        arrays += [_float_array(columns[k], np.float32, pa.float32()) for k in spokes]
        schema = pa.schema(
            [*FRAME_SCHEMA, *(pa.field(k, pa.float32()) for k in spokes)]
        )
    return pa.Table.from_arrays(arrays, schema=schema)


def write_frames(
    path: str | Path, columns: dict[str, np.ndarray], metadata: dict
) -> None:
    """Write a frame table with the storage of §3.2 and the metadata of §3.3.

    Float columns may hold NaN for missing values, which are stored as null.
    The LED columns and ``led_x``, ``led_y`` may be omitted, or given as floats
    with NaN, for a run without a sync LED. The spoke columns are written,
    after the others, only when given.

    Args:
        path: Output file; its stem must equal ``metadata["run"]``.
        columns: Column name to 1-D array, as in §3.1.
        metadata: The §3.3 metadata.

    Raises:
        ValueError: If a column or a metadata value is missing or invalid.
    """
    path = Path(path)
    validate_metadata(metadata)
    if metadata["run"] != path.stem:
        raise ValueError(
            f"metadata key 'run' is {metadata['run']!r}, but the file stem is "
            f"{path.stem!r}"
        )
    table = _build_table(columns)
    encodings = {"frame": "DELTA_BINARY_PACKED", "pts": "DELTA_BINARY_PACKED"}
    encodings.update(
        {
            name: "BYTE_STREAM_SPLIT"
            for name in FLOAT32_COLUMNS + FLOAT16_COLUMNS + SPOKE_COLUMNS
            if name in table.schema.names
        }
    )
    dictionary = [name for name in table.schema.names if name not in encodings]
    with pq.ParquetWriter(
        path,
        table.schema,
        compression="zstd",
        compression_level=9,
        use_dictionary=dictionary,
        column_encoding=encodings,
        # Keep the one metadata key of §3.3; Parquet's own logical types
        # restore uint32 and float16 without the Arrow schema copy. Without
        # the copy, pyarrow drops table metadata, so the key is added here.
        store_schema=False,
    ) as writer:
        writer.add_key_value_metadata(
            {METADATA_KEY: json.dumps(metadata, ensure_ascii=False)}
        )
        writer.write_table(table, row_group_size=max(table.num_rows, 1))


def read_frames(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read a frame table.

    Args:
        path: The Parquet file.

    Returns:
        Column name to numpy array, and the §3.3 metadata. Counts are uint32,
        ``frame`` uint32, ``pts`` int64, and float columns keep their stored
        precision with NaN for null. An LED column is uint8 when it has no
        null, and float32 with NaN otherwise. The spoke columns are returned
        only when the table has them.

    Raises:
        ValueError: If the metadata is missing or invalid, or the columns do
            not match §3.1.
    """
    table = pq.read_table(path)
    raw_meta = (table.schema.metadata or {}).get(METADATA_KEY.encode())
    if raw_meta is None:
        raise ValueError(f"{path} has no {METADATA_KEY!r} metadata key")
    metadata = json.loads(raw_meta)
    validate_metadata(metadata)
    spokes = table.schema.names == FRAME_SCHEMA.names + SPOKE_COLUMNS
    if table.schema.names != FRAME_SCHEMA.names and not spokes:
        raise ValueError(
            f"{path} columns differ from schema §3.1: {table.schema.names}"
        )
    for field in FRAME_SCHEMA:
        stored = table.schema.field(field.name).type
        if stored != field.type:
            raise ValueError(
                f"{path} column {field.name!r} is {stored}, expected {field.type}"
            )
    out: dict[str, np.ndarray] = {}
    for field in FRAME_SCHEMA:
        col = table.column(field.name)
        dtype = np.dtype(field.type.to_pandas_dtype())
        if field.name in LED_COLUMNS and col.null_count:
            filled = col.cast(pa.float32()).fill_null(float("nan"))
            out[field.name] = filled.to_numpy().astype(np.float32)
        elif dtype.kind == "f":
            out[field.name] = col.fill_null(float("nan")).to_numpy().astype(dtype)
        else:
            if col.null_count:
                raise ValueError(f"{path} column {field.name!r} has nulls")
            out[field.name] = col.to_numpy().astype(dtype)
    for name in SPOKE_COLUMNS if spokes else ():
        col = table.column(name)
        if col.type != pa.float32():
            raise ValueError(f"{path} column {name!r} is {col.type}, expected float")
        out[name] = col.fill_null(float("nan")).to_numpy().astype(np.float32)
    return out, metadata
