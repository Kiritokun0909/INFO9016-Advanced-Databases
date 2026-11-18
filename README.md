# NYC Yellow Taxi Dataset Preparation and Format Conversion

This project uses **PyArrow** to prepare reproducible datasets from the [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and compare three analytical file formats: **Parquet**, **ORC**, and **Arrow IPC (file format)**. It includes three standalone scripts, configured by editing Python variables rather than command-line arguments.

## 1. Requirements

- Python 3.10 or newer (recommended)
- `pyarrow`
- `tzdata` (required for ORC conversion on Windows; install it on all platforms for a consistent environment)
- Enough free disk space for the original files and generated datasets

Install dependencies in the Python environment used to run the scripts:

```bash
python -m pip install pyarrow tzdata
```

> **Windows / PyCharm:** Make sure the `python` command points to the same virtual environment as your PyCharm project. `convert_formats.py` sets `TZDIR` automatically on Windows using the installed `tzdata` package, before importing PyArrow.

## 2. Project structure

Recommended layout (folders in the scripts' default configuration are shown):

```text
project/
├── create_dataset.py
├── convert_formats.py
├── inspect_file.py
├── README.md
├── dataset/                    # Original NYC TLC monthly Parquet files
│   ├── yellow_tripdata_2025-01.parquet
│   ├── yellow_tripdata_2025-02.parquet
│   ├── yellow_tripdata_2025-03.parquet
│   ├── yellow_tripdata_2025-04.parquet
│   └── yellow_tripdata_2025-05.parquet
├── prepared_datasets/          # Created by create_dataset.py
│   ├── small.parquet
│   ├── medium.parquet
│   └── large.parquet
└── convert_dataset/            # Created by convert_formats.py
    ├── small_none.parquet
    ├── small_none.orc
    ├── small_none.arrow
    ├── small_zstd.parquet
    ├── small_zstd.orc
    ├── small_zstd.arrow
    ├── ...                       # Same six combinations for medium and large
    └── manifest.csv
```

Download the **Yellow Taxi Trip Records** monthly Parquet files for 2025 from the NYC TLC page and place them in `dataset/`. The current `create_dataset.py` configuration lists **January–April 2025**; edit that list if you have a different selection of months or need more rows.

All paths are relative to your **current working directory**. The simplest approach is to run the scripts from the project root.

## 3. Step 1 — Create the three dataset sizes

Open `create_dataset.py` and edit the **USER CONFIGURATION** section:

```python
INPUT_DIR = Path("dataset")
INPUT_FILES = [
    INPUT_DIR / f"yellow_tripdata_2025-{month:02d}.parquet"
    for month in range(1, 5)    # depend on which months from to in folder dataset
]
OUTPUT_DIR = Path("prepared_datasets")

DATASET_SIZES = {
    "small": 100_000,
    "medium": 1_000_000,
    "large": 10_000_000,
}

COMPRESSION = "zstd"
ZSTD_LEVEL = 3
BATCH_SIZE = 65_536
ROW_GROUP_SIZE = 65_536
```

Run:

```bash
python create_dataset.py
```

The script streams batches from the input files in the order listed, rather than loading all source months into memory. It creates:

| File | Rows | Compression |
|---|---:|---|
| `prepared_datasets/small.parquet` | 100,000 | ZSTD (level 3) |
| `prepared_datasets/medium.parquet` | 1,000,000 | ZSTD (level 3) |
| `prepared_datasets/large.parquet` | 10,000,000 | ZSTD (level 3) |

The datasets are **nested prefixes**, not independent or random samples: `small` contains the first 100,000 source rows, `medium` contains the first 1,000,000, and `large` contains the first 10,000,000. The script stops once it has written the largest target. It verifies each output's row count, Arrow schema, and Parquet column-chunk codec metadata.

**Important:** Your selected source files must collectively contain at least 10 million rows. The script raises an error if they do not. It also checks for missing/duplicate input files and rejects unsafe type casts between monthly schemas.

## 4. Step 2 — Convert to Parquet, ORC, and Arrow IPC

Open `convert_formats.py` and edit its **USER CONFIGURATION** section:

```python
INPUT_DIR = Path("prepared_datasets")
OUTPUT_DIR = Path("convert_dataset")
FILE_PATTERN = "*.parquet"
CODECS = ["none", "zstd"]
FORMATS = ["parquet", "orc", "arrow"]
BATCH_SIZE = 65_536
ZSTD_LEVEL = 3
VERIFY_OUTPUTS = True
```

Run:

```bash
python convert_formats.py
```

For each input file (`small.parquet`, `medium.parquet`, and `large.parquet`), the script reads **one PyArrow Table** and writes all six combinations:

| Output suffix | File format | Compression |
|---|---|---|
| `_none.parquet` | Parquet | None |
| `_zstd.parquet` | Parquet | ZSTD |
| `_none.orc` | ORC | None |
| `_zstd.orc` | ORC | ZSTD |
| `_none.arrow` | Arrow IPC file | None |
| `_zstd.arrow` | Arrow IPC file | ZSTD |

With all default options, this produces **18 converted files** (3 dataset sizes × 3 formats × 2 codecs), plus `convert_dataset/manifest.csv`.

The manifest records the input/output filename, format, codec, row/column counts, size in bytes, writing time in seconds, verification flag, and PyArrow version. The measured write time covers writing the file, **not** verification and not necessarily the source read time.

For ZSTD, `ZSTD_LEVEL = 3` applies **only to Parquet**. The PyArrow ORC and Arrow IPC writer APIs used by the script select ZSTD but do not expose an equivalent level option. Also, Parquet row groups, ORC stripes, and Arrow IPC record batches have different physical layouts; selecting the same compression codec does not make the file internals identical.

### Output verification

With `VERIFY_OUTPUTS = True`, each converted file is read back and compared with the original in-memory Arrow Table. Verification checks row counts, column names/order, types, and all data values. For ORC, the verifier permits only specific known type representation changes, such as `timestamp[us]` → `timestamp[ns]` and `large_string` → `string`, then **safely casts back** for comparison. Any unsupported type change or mismatched data raises an error; the failed output file is removed.

**Memory note:** This script loads each entire prepared dataset into memory one at a time. The 10-million-row dataset may require substantial RAM, particularly during verification. The initial dataset-creation script is streaming, but the format-conversion script is not.

## 5. Step 3 — Inspect a file

Open `inspect_file.py` and change:

```python
FILE_PATH = Path("prepared_datasets/small.parquet")
FILE_FORMAT = "auto"  # auto, parquet, orc, arrow
```

Run:

```bash
python inspect_file.py
```

Change `FILE_PATH` to inspect a converted file instead, for example:

```python
FILE_PATH = Path("convert_dataset/large_zstd.orc")
```

The script reports the file's size, number of rows/columns, schema, and format-specific storage metadata:

| Format | Additional information |
|---|---|
| Parquet | Row-group count, column-chunk compression codecs, encodings |
| ORC | Stripe count and file-level compression kind |
| Arrow IPC | Record-batch count; no simple codec property exposed by the script |

For Arrow IPC files produced by the conversion script, check the filename and `manifest.csv` for the **configured** codec. This is not independent metadata verification of the codec inside the IPC file.

## 6. Suggested benchmark workflow

1. Download the original 2025 monthly Yellow Taxi Parquet files.
2. Run `python create_dataset.py` to make fixed-size, ZSTD-compressed source datasets.
3. Run `python convert_formats.py` to generate all format/codec combinations from the same per-size Arrow Table.
4. Run `python inspect_file.py` on selected outputs and retain `manifest.csv` as conversion metadata.
5. Benchmark file size, read/I/O cost, and query execution time using a consistent environment and methodology.

### Interpreting fairness

- **Logical comparability:** The converter verifies identical row counts, column order, and data values after round-trip reading (with limited safe ORC normalization).
- **Not identical physical storage:** Row groups, stripes, record batches, encodings, and compression options differ by format. Document these as part of the comparison.
- **Not a random sample:** Because the three dataset sizes are prefixes, they can have different distributions from a random sample of the full year. Record the input months and selection method in your report.
- **Reproducibility:** Record the Python and PyArrow versions, source month filenames, configuration values, hardware, and benchmark settings. Use `manifest.csv` for generated file sizes and write durations.

## 7. Common errors

| Error | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'pyarrow'` | Install it in the active environment: `python -m pip install pyarrow`. |
| `No input files matched ...` | Check `INPUT_DIR`, `FILE_PATTERN`, and the current working directory. |
| `Not enough input rows` | Add more monthly Parquet files to `INPUT_FILES`. |
| `Time zone file /usr/share/zoneinfo/GMT does not exist` | Install `tzdata` in the active Windows environment; `convert_formats.py` configures `TZDIR`. |
| `Schema differs from first month` | Monthly input schemas differ. Normalize them explicitly before running the converter. |
| Out-of-memory during conversion | `convert_formats.py` materializes each input Parquet file as an Arrow Table; reduce input size or implement streaming conversion. |

## 8. Source and documentation

- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [PyArrow Parquet documentation](https://arrow.apache.org/docs/python/parquet.html)
- [PyArrow ORC documentation](https://arrow.apache.org/docs/python/orc.html)
- [PyArrow IPC documentation](https://arrow.apache.org/docs/python/ipc.html)

**Scripts covered:** `create_dataset.py`, `convert_formats.py`, and `inspect_file.py` (as uploaded).
