"""Create reproducible small/medium/large NYC Yellow Taxi Parquet datasets
by seeded random sampling from the combined January-May 2024 data.

Install: python -m pip install pyarrow numpy
Run:     python create_dataset.py

Sampling design
---------------
1. The five monthly files are treated as one virtual table of T rows.
   A row's global id is its position in the concatenation Jan, Feb, ..., May.
2. One random permutation of the ids 0..T-1 is drawn with NumPy's default
   generator (PCG64) seeded with SEED.
3. large  = the first 10,000,000 ids of that permutation
   medium = the first  1,000,000 ids
   small  = the first    100,000 ids
   Hence small is a subset of medium, medium is a subset of large, and each
   one is a uniform random sample without replacement from all T rows.
4. Selected rows are written in their original (file) order, so the physical
   row layout does not depend on the order in which ids were drawn.

Input files are streamed batch by batch; only the selected ids and one batch
are held in memory. The selected ids are saved to selection.npz and the run
parameters to selection_manifest.json, so the sample can be audited or rebuilt.
Same SEED + same input files + same NumPy version => identical datasets.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ===================== USER CONFIGURATION =====================
INPUT_DIR = Path("original_dataset")
INPUT_FILES = [
    INPUT_DIR / f"yellow_tripdata_2024-{month:02d}.parquet"
    for month in range(1, 6)  # January to May 2024
]
OUTPUT_DIR = Path("prepared_datasets")

DATASET_SIZES = {
    "small": 100_000,
    "medium": 1_000_000,
    "large": 10_000_000,
}

SEED = 42  # Fixed seed: do not change between runs of the experiment

COMPRESSION = "zstd"
# Default zstd level
# Ref: https://docs.aws.amazon.com/athena/latest/ug/compression-support-zstd-levels.html
ZSTD_LEVEL = 3
BATCH_SIZE = 65_536      # Maximum rows read per input batch
ROW_GROUP_SIZE = 65_536  # Rows per output row group (last one may be smaller)
# ==============================================================


class RowGroupBuffer:
    """Collect sampled rows and write them in full row groups.

    Sampling keeps only a fraction of each input batch, so writing each
    fragment directly would create many tiny row groups. This buffer writes
    groups of exactly ROW_GROUP_SIZE rows, except possibly the last one.
    """

    def __init__(self, writer: pq.ParquetWriter, rows_per_group: int):
        self.writer = writer
        self.rows_per_group = rows_per_group
        self.parts: list[pa.Table] = []
        self.buffered = 0

    def add(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        self.parts.append(table)
        self.buffered += table.num_rows
        while self.buffered >= self.rows_per_group:
            combined = pa.concat_tables(self.parts)
            self.writer.write_table(combined.slice(0, self.rows_per_group),
                                    row_group_size=self.rows_per_group)
            rest = combined.slice(self.rows_per_group)
            self.parts = [rest] if rest.num_rows else []
            self.buffered = rest.num_rows

    def flush(self) -> None:
        if self.buffered:
            self.writer.write_table(pa.concat_tables(self.parts),
                                    row_group_size=self.rows_per_group)
        self.parts, self.buffered = [], 0


def draw_selection(total_rows: int) -> dict[str, np.ndarray]:
    """Return sorted global row ids for each dataset size (nested subsets)."""
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(total_rows)
    selection = {
        name: np.sort(permutation[:size]).astype(np.int64)
        for name, size in DATASET_SIZES.items()
    }
    del permutation
    return selection


def check_nested(selection: dict[str, np.ndarray]) -> None:
    """Assert that every smaller dataset is a subset of the next larger one."""
    names = sorted(selection, key=lambda n: DATASET_SIZES[n])
    for smaller, larger in zip(names, names[1:]):
        if not np.isin(selection[smaller], selection[larger],
                       assume_unique=True).all():
            raise AssertionError(f"{smaller} is not a subset of {larger}")


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

    # The first input schema is authoritative. Safe casting supports e.g.
    # compatible month-to-month types while rejecting lossy conversions.
    schema = pq.ParquetFile(INPUT_FILES[0]).schema_arrow
    input_rows = {}
    for path in INPUT_FILES:
        pf = pq.ParquetFile(path)
        # Do not silently change column names/order or meanings.
        if pf.schema_arrow.names != schema.names:
            raise ValueError(f"Column names/order differ in {path}")
        input_rows[path.name] = pf.metadata.num_rows
    total_rows = sum(input_rows.values())

    max_rows = max(DATASET_SIZES.values())
    if max_rows > total_rows:
        raise ValueError(f"Requested {max_rows:,} rows but inputs contain only {total_rows:,}")
    print(f"Combined input: {total_rows:,} rows from {len(INPUT_FILES)} files")

    selection = draw_selection(total_rows)
    check_nested(selection)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = {name: OUTPUT_DIR / f"{name}.parquet" for name in DATASET_SIZES}
    counts = {name: 0 for name in DATASET_SIZES}
    writers, buffers = {}, {}
    completed = False
    try:
        for name, path in output_paths.items():
            writers[name] = pq.ParquetWriter(
                where=path,
                schema=schema,
                compression=COMPRESSION,
                compression_level=ZSTD_LEVEL,
            )
            buffers[name] = RowGroupBuffer(writers[name], ROW_GROUP_SIZE)

        offset = 0  # global id of the first row in the current batch
        for input_path in INPUT_FILES:
            print(f"Reading {input_path.name} ...")
            pf = pq.ParquetFile(input_path)
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, use_threads=True):
                n = batch.num_rows
                table = pa.Table.from_batches([batch])
                if not table.schema.equals(schema, check_metadata=False):
                    try:
                        table = table.cast(schema, safe=True)
                    except (pa.ArrowException, ValueError) as exc:
                        raise ValueError(f"Incompatible schema in {input_path}: {exc}") from exc

                for name, ids in selection.items():
                    lo = np.searchsorted(ids, offset, side="left")
                    hi = np.searchsorted(ids, offset + n, side="left")
                    if hi > lo:
                        local = pa.array(ids[lo:hi] - offset)
                        buffers[name].add(table.take(local))
                        counts[name] += hi - lo
                offset += n

        if offset != total_rows:
            raise ValueError(f"Read {offset:,} rows but metadata reported {total_rows:,}")
        for buffer in buffers.values():
            buffer.flush()
        if any(counts[name] != target for name, target in DATASET_SIZES.items()):
            details = ", ".join(f"{name}: {counts[name]}/{target}"
                                for name, target in DATASET_SIZES.items())
            raise ValueError(f"Row count mismatch. Generated: {details}")
        completed = True
    finally:
        for writer in writers.values():
            writer.close()
        # If generation fails, don't leave misleading partial benchmark files.
        if not completed:
            for path in output_paths.values():
                path.unlink(missing_ok=True)

    # Save the exact selection and the parameters needed to reproduce it.
    np.savez_compressed(OUTPUT_DIR / "selection.npz", **selection)
    manifest = {
        "seed": SEED,
        "method": "first N ids of one numpy.random.default_rng(SEED).permutation(T); "
                  "rows written in original file order",
        "input_files": input_rows,
        "total_input_rows": total_rows,
        "dataset_sizes": DATASET_SIZES,
        "compression": COMPRESSION,
        "zstd_level": ZSTD_LEVEL,
        "row_group_size": ROW_GROUP_SIZE,
        "numpy_version": np.__version__,
        "pyarrow_version": pa.__version__,
    }
    (OUTPUT_DIR / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nOutput verification:")
    for name, path in output_paths.items():
        metadata = pq.read_metadata(path)
        if metadata.num_rows != DATASET_SIZES[name]:
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
        print(f"  {path}: {metadata.num_rows:,} rows | {metadata.num_row_groups} row groups "
              f"| {size_mib:.2f} MiB | ZSTD verified")

    print(f"\nDone. Random sample (seed={SEED}); small ⊂ medium ⊂ large verified.")


if __name__ == "__main__":
    main()