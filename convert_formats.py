#!/usr/bin/env python3
"""Convert monthly NYC TLC Parquet files to equivalent Parquet, ORC and Arrow IPC files.

Each input month is materialized ONCE as a PyArrow Table and written into all
requested codecs/formats. Months are processed sequentially to limit peak RAM.
"""
import csv
import os
from pathlib import Path
import time

# The ORC C++ writer needs the IANA time-zone database for timestamp fields.
# Windows pip installations may not discover this database automatically.
# Install the dependency once using: python -m pip install tzdata
if os.name == "nt":
    try:
        import tzdata
    except ImportError as exc:
        raise RuntimeError(
            "ORC conversion on Windows requires tzdata. "
            "Install it in this environment: python -m pip install tzdata"
        ) from exc

    zoneinfo_dir = Path(tzdata.__file__).resolve().parent / "zoneinfo"
    if not (zoneinfo_dir / "GMT").is_file():
        raise RuntimeError(f"tzdata installation is missing GMT: {zoneinfo_dir}")
    os.environ["TZDIR"] = str(zoneinfo_dir)

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.orc as orc
import pyarrow.ipc as ipc


# ====================== USER CONFIGURATION ======================
# Change these values, then run: python convert_formats.py
INPUT_DIR = Path("prepared_datasets")
OUTPUT_DIR = Path("convert_dataset")
# FILE_PATTERN = "yellow_tripdata_2025-*.parquet"
FILE_PATTERN = "*.parquet"
CODECS = ["none", "zstd"]       # Either or both
FORMATS = ["parquet", "orc", "arrow"]  # Any subset
BATCH_SIZE = 65_536             # Parquet row-group / Arrow record-batch rows
ZSTD_LEVEL = 3                  # Applies to Parquet only
VERIFY_OUTPUTS = True          # Read back each output and compare all values
# ================================================================


def write_output(table: pa.Table, destination: Path, fmt: str, codec: str,
                 batch_size: int, zstd_level: int) -> None:
    codec_arg = None if codec == 'none' else 'zstd'
    if fmt == 'parquet':
        pq.write_table(table, destination,
                       compression=codec_arg,
                       compression_level=(zstd_level if codec == 'zstd' else None),
                       row_group_size=batch_size,
                       use_dictionary=True,
                       version='2.6')
    elif fmt == 'orc':
        # ORC exposes codec selection but no equivalent ZSTD-level writer option.
        orc.write_table(table, destination,
                        compression=('uncompressed' if codec == 'none' else 'zstd'))
    elif fmt == 'arrow':
        options = ipc.IpcWriteOptions(compression=codec_arg)
        with pa.OSFile(str(destination), 'wb') as sink:
            with ipc.new_file(sink, table.schema, options=options) as writer:
                for batch in table.to_batches(max_chunksize=batch_size):
                    writer.write_batch(batch)
    else:
        raise ValueError(fmt)


def read_output(path: Path, fmt: str) -> pa.Table:
    if fmt == 'parquet':
        return pq.read_table(path)
    if fmt == 'orc':
        return orc.read_table(path)
    with pa.memory_map(str(path), 'r') as source:
        return ipc.open_file(source).read_all()


def verify(source: pa.Table, output: Path, fmt: str) -> None:
    restored = read_output(output, fmt)
    if source.num_rows != restored.num_rows:
        raise ValueError(f"Row count changed: {output}")
    if source.column_names != restored.column_names:
        raise ValueError(f"Column names/order changed: {output}")

    source_schema = source.schema.remove_metadata()
    restored_schema = restored.schema.remove_metadata()

    if not source_schema.equals(restored_schema, check_metadata=False):
        if fmt != "orc":
            raise ValueError(f"Unexpected schema change: {output}\n"
                             f"input: {source_schema}\noutput: {restored_schema}")

        # PyArrow's ORC reader can represent timestamp[us] as timestamp[ns]
        # and large_string as string. Accept ONLY these known differences.
        # All other schema differences are errors, including nullable changes.
        for original, actual in zip(source_schema, restored_schema):
            allowed_timestamp = (
                pa.types.is_timestamp(original.type)
                and pa.types.is_timestamp(actual.type)
                and original.type.unit == "us"
                and actual.type.unit == "ns"
                and original.type.tz == actual.type.tz
            )
            allowed_string = (
                pa.types.is_large_string(original.type)
                and pa.types.is_string(actual.type)
            )
            if (original.name != actual.name
                    or original.nullable != actual.nullable
                    or not (original.type == actual.type
                            or allowed_timestamp or allowed_string)):
                raise ValueError(
                    f"Unsupported ORC schema change in {original.name}: "
                    f"{original.type} -> {actual.type} ({output})"
                )

        # Safe casting timestamp[ns] -> timestamp[us] raises if any
        # nanosecond precision would be lost; strings retain their content.
        # The original schema is used ONLY for verification, not rewriting.
        restored = restored.cast(source_schema, safe=True)
        print(f"  ORC schema normalized for comparison: {output.name}", flush=True)

    if not source.equals(restored, check_metadata=False):
        raise ValueError(f"Data differs after roundtrip: {output}")


def main():
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive")
    if not CODECS or any(c not in ("none", "zstd") for c in CODECS):
        raise ValueError("CODECS must contain 'none', 'zstd', or both")
    if not FORMATS or any(f not in ("parquet", "orc", "arrow") for f in FORMATS):
        raise ValueError("FORMATS must contain parquet, orc, and/or arrow")
    files = sorted(INPUT_DIR.glob(FILE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No input files matched {INPUT_DIR / FILE_PATTERN}")
    if "zstd" in CODECS and not pa.Codec.is_available("zstd"):
        raise RuntimeError("This PyArrow build does not support ZSTD")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / "manifest.csv"
    expected_schema = None
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        out = csv.DictWriter(handle, fieldnames=[
            "input", "output", "format", "codec", "rows", "columns", "bytes",
            "write_seconds", "verified", "pyarrow_version"
        ])
        out.writeheader()
        for source in files:
            print(f"\nReading {source.name}", flush=True)
            table = pq.read_table(source)
            schema = table.schema.remove_metadata()
            if expected_schema is None:
                expected_schema = schema
            elif not schema.equals(expected_schema, check_metadata=False):
                raise ValueError(
                    f"Schema differs from first month: {source.name}. "
                    "Normalize month schemas explicitly before benchmarking."
                )
            print(f"  {table.num_rows:,} rows, {table.num_columns} columns", flush=True)
            for codec in CODECS:
                for fmt in FORMATS:
                    extension = {"parquet": ".parquet", "orc": ".orc", "arrow": ".arrow"}[fmt]
                    destination = OUTPUT_DIR / f"{source.stem}_{codec}{extension}"
                    start = time.perf_counter()
                    try:
                        write_output(table, destination, fmt, codec, BATCH_SIZE, ZSTD_LEVEL)
                        duration = time.perf_counter() - start
                        if VERIFY_OUTPUTS:
                            verify(table, destination, fmt)
                    except Exception:
                        destination.unlink(missing_ok=True)
                        raise
                    result = dict(
                        input=source.name, output=destination.name, format=fmt,
                        codec=codec, rows=table.num_rows, columns=table.num_columns,
                        bytes=destination.stat().st_size,
                        write_seconds=f"{duration:.4f}",
                        verified=("yes" if VERIFY_OUTPUTS else "no"),
                        pyarrow_version=pa.__version__,
                    )
                    out.writerow(result)
                    handle.flush()
                    print(
                        f"  {destination.name}: {result['bytes'] / 2**20:.2f} MiB, "
                        f"{duration:.2f}s, verified={result['verified']}",
                        flush=True,
                    )
            del table
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
