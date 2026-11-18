"""Create reproducible small/medium/large NYC Yellow Taxi Parquet datasets.

Install: python -m pip install pyarrow
Run:     python create_dataset_sizes.py

Rows are selected in INPUT_FILES order, preserving the original row order.
The three outputs are nested prefixes: small is contained in medium, and
medium is contained in large. Input files are streamed, not fully loaded.
"""

from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# ===================== USER CONFIGURATION =====================
INPUT_DIR = Path("dataset")
INPUT_FILES = [
    INPUT_DIR / f"yellow_tripdata_2025-{month:02d}.parquet"
    for month in range(1, 5)
]
OUTPUT_DIR = Path("prepared_datasets")

DATASET_SIZES = {
    "small": 100_000,
    "medium": 1_000_000,
    "large": 10_000_000,
}

COMPRESSION = "zstd"
# Default zstd level
# Ref: https://docs.aws.amazon.com/athena/latest/ug/compression-support-zstd-levels.html
ZSTD_LEVEL = 3
BATCH_SIZE = 65_536   # Maximum rows read per input batch
ROW_GROUP_SIZE = 65_536  # Target max rows per output row group
# ==============================================================


def main() -> None:
    if not INPUT_FILES:
        raise ValueError("INPUT_FILES is empty")
    if not DATASET_SIZES or any(n <= 0 for n in DATASET_SIZES.values()):
        raise ValueError("DATASET_SIZES must contain positive row counts")

    missing = [str(path) for path in INPUT_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input Parquet files:\n" + "\n".join(missing))

    # Reject accidental duplicate inputs (which would duplicate rows).
    resolved = [path.resolve() for path in INPUT_FILES]
    if len(set(resolved)) != len(resolved):
        raise ValueError("INPUT_FILES contains duplicate paths")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = dict(DATASET_SIZES)
    counts = {name: 0 for name in targets}
    output_paths = {name: OUTPUT_DIR / f"{name}.parquet" for name in targets}
    max_rows = max(targets.values())

    # The first input schema is authoritative. Safe casting supports e.g.
    # compatible month-to-month types while rejecting lossy conversions.
    schema = pq.ParquetFile(INPUT_FILES[0]).schema_arrow
    writers = {}
    completed = False
    try:
        for name, path in output_paths.items():
            writers[name] = pq.ParquetWriter(
                where=path,
                schema=schema,
                compression=COMPRESSION,
                compression_level=ZSTD_LEVEL,
            )

        for input_path in INPUT_FILES:
            if counts[max(targets, key=targets.get)] >= max_rows:
                break
            print(f"Reading {input_path.name} ...")
            pf = pq.ParquetFile(input_path)
            # Do not silently change column names/order or meanings.
            if pf.schema_arrow.names != schema.names:
                raise ValueError(f"Column names/order differ in {input_path}")

            for batch in pf.iter_batches(batch_size=BATCH_SIZE, use_threads=True):
                table = pa.Table.from_batches([batch])
                if not table.schema.equals(schema, check_metadata=False):
                    try:
                        table = table.cast(schema, safe=True)
                    except (pa.ArrowException, ValueError) as exc:
                        raise ValueError(f"Incompatible schema in {input_path}: {exc}") from exc

                for name, target_rows in targets.items():
                    needed = target_rows - counts[name]
                    if needed <= 0:
                        continue
                    take = min(needed, table.num_rows)
                    if take:
                        writers[name].write_table(table.slice(0, take), row_group_size=ROW_GROUP_SIZE)
                        counts[name] += take

                if all(counts[name] == target for name, target in targets.items()):
                    break

        if any(counts[name] != target for name, target in targets.items()):
            details = ", ".join(f"{name}: {counts[name]}/{target}" for name, target in targets.items())
            raise ValueError(f"Not enough input rows. Generated: {details}")
        completed = True
    finally:
        for writer in writers.values():
            writer.close()
        # If generation fails, don't leave misleading partial benchmark files.
        if not completed:
            for path in output_paths.values():
                path.unlink(missing_ok=True)

    print("\nOutput verification:")
    for name, path in output_paths.items():
        metadata = pq.read_metadata(path)
        if metadata.num_rows != targets[name]:
            raise AssertionError(f"Incorrect row count in {path}: {metadata.num_rows}")
        if not pq.ParquetFile(path).schema_arrow.equals(schema, check_metadata=False):
            raise AssertionError(f"Schema mismatch in {path}")
        codecs = {
            metadata.row_group(i).column(j).compression
            for i in range(metadata.num_row_groups)
            for j in range(metadata.num_columns)
        }
        if codecs != {"ZSTD"}:
            raise AssertionError(f"Expected ZSTD in {path}, got {codecs}")
        size_mib = path.stat().st_size / (1024 * 1024)
        print(f"  {path}: {metadata.num_rows:,} rows | {size_mib:.2f} MiB | ZSTD verified")

    print("\nDone. Datasets use the first N rows in the configured input order.")


if __name__ == "__main__":
    main()
