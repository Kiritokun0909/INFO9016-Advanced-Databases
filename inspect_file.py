#!/usr/bin/env python3
"""Inspect a Parquet, ORC or Arrow IPC *file* without loading all data."""
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.orc as orc
import pyarrow.ipc as ipc


# ====================== USER CONFIGURATION ======================
# Change these values, then run: python inspect_file.py
FILE_PATH = Path("dataset/yellow_tripdata_2025-01.parquet")
FILE_FORMAT = "auto"  # "auto", "parquet", "orc", or "arrow"
# ================================================================


def inspect(path: Path, fmt: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if fmt == 'auto':
        fmt = {'.parquet': 'parquet', '.orc': 'orc', '.arrow': 'arrow',
               '.ipc': 'arrow', '.feather': 'arrow'}.get(path.suffix.lower())
        if fmt is None:
            raise ValueError('Unknown extension: set FILE_FORMAT to parquet, orc, or arrow')

    print(f'File: {path.resolve()}')
    print(f'Format: {fmt.upper()}')
    print(f'Size: {path.stat().st_size:,} bytes ({path.stat().st_size / 2**20:.2f} MiB)')

    if fmt == 'parquet':
        f = pq.ParquetFile(path)
        md = f.metadata
        schema = f.schema_arrow
        print(f'Rows: {md.num_rows:,}\nColumns: {len(schema)}\nRow groups: {md.num_row_groups}')
        print('\nColumn chunk compression (by row group):')
        for index, field in enumerate(schema):
            # Path-based match works for the taxi dataset's flat columns.
            codecs = set()
            encodings = set()
            for rg in range(md.num_row_groups):
                group = md.row_group(rg)
                for col in range(group.num_columns):
                    chunk = group.column(col)
                    if chunk.path_in_schema == field.name:
                        codecs.add(chunk.compression)
                        encodings.update(chunk.encodings)
            print(f'  {field.name}: codec={", ".join(sorted(codecs)) or "n/a"}; '
                  f'encodings={", ".join(sorted(encodings)) or "n/a"}')
        if any(pa.types.is_struct(f.type) or pa.types.is_list(f.type) for f in schema):
            print('Note: nested fields have leaf-level column chunks; inspect raw paths for detail.')
    elif fmt == 'orc':
        f = orc.ORCFile(path)
        schema = f.schema
        print(f'Rows: {f.nrows:,}\nColumns: {len(schema)}\nStripes: {f.nstripes}')
        print(f'Compression (file-level): {f.compression}')
        print('Per-column codec: not exposed; ORC uses a file-level compression kind.')
    elif fmt == 'arrow':
        with pa.memory_map(str(path), 'r') as source:
            f = ipc.open_file(source)
            schema = f.schema
            rows = sum(f.get_batch(i).num_rows for i in range(f.num_record_batches))
            print(f'Rows: {rows:,}\nColumns: {len(schema)}'
                  f'\nRecord batches: {f.num_record_batches}')
        print('Compression: Arrow IPC can compress record-batch buffers, but its Python '
              'file reader does not expose a simple per-column codec property.')
        print('For files from convert_formats.py, use the codec in their filename/log.')
    else:
        raise ValueError(f'Unsupported format: {fmt}')

    print('\nSchema:')
    for field in schema:
        print(f'  {field.name}: {field.type} (nullable={field.nullable})')


if __name__ == '__main__':
    inspect(FILE_PATH, FILE_FORMAT)
