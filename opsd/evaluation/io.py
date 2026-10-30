"""Atomic files and Parquet inputs shared by benchmark adapters."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: str | Path, value: object) -> None:
    write_text(Path(path), json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_parquet(rows: list[dict], packing: dict, parquet: Path) -> None:
    import pyarrow as arrow
    import pyarrow.parquet as pq
    from datasets import Features, Image, Value

    match = {key: Value("string") for key in ("dataset", "split", "sample_id", "relation", "revision")}
    features = Features({"image": Image(), "prompt": Value("string"),
                         "actions": [[Value("string")]], "source": {
                             "sample_id": Value("string"), "dataset": Value("string"),
                             "asset_id": Value("string"), "url": Value("string"),
                             "revision": Value("string"), "category": Value("string"),
                             "dataset_matches": [match], "benchmark_matches": [match],
                             "camera_task_template": {key: Value("string") for key in
                                                      ("benchmark", "split", "sample_id", "relation")}}})
    schema = features.arrow_schema
    metadata = dict(schema.metadata or {})
    metadata[b"world_model"] = json.dumps(packing, allow_nan=False).encode("utf-8")
    parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary = parquet.with_name(f".{parquet.name}.tmp")
    try:
        with pq.ParquetWriter(temporary, schema.with_metadata(metadata), compression="zstd") as writer:
            for row in rows:
                writer.write_table(arrow.Table.from_pylist([row], schema=schema))
        with pq.ParquetFile(temporary) as restored:
            if (restored.metadata.num_rows != len(rows)
                    or restored.schema_arrow.metadata != metadata):
                raise ValueError("Parquet round-trip changed the records or packing metadata")
            for row, actual in zip(rows,
                                        restored.iter_batches(batch_size=1, use_threads=False), strict=True):
                expected = arrow.Table.from_pylist([row], schema=schema).to_pylist()
                if expected != actual.to_pylist():
                    raise ValueError("Parquet round-trip changed a record")
        temporary.replace(parquet)
    finally:
        temporary.unlink(missing_ok=True)
