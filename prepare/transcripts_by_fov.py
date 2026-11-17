"""Split the CosMx breast transcript table into one parquet file per imaging field (cell_ID, global x/y in pixels, target),
used to draw transcript positions in the case figures. Writes data/breast/tx_by_fov/fov###.parquet."""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from graticcs.paths import BREAST, RAW

OUT = BREAST / "tx_by_fov"; OUT.mkdir(exist_ok=True)
writers = {}
for chunk in pd.read_csv(RAW / "cosmx_breast" / "BreastCancer_tx_file.csv", usecols=["fov", "cell_ID", "x_global_px", "y_global_px", "target"],
                         dtype={"fov": "int16", "cell_ID": "int32", "x_global_px": "float32", "y_global_px": "float32", "target": "string"},
                         chunksize=5_000_000):
    for f, g in chunk.groupby("fov"):
        t = pa.Table.from_pandas(g.drop(columns="fov").reset_index(drop=True), preserve_index=False)
        if f not in writers:
            writers[f] = pq.ParquetWriter(OUT / f"fov{int(f):03d}.parquet", t.schema, compression="zstd")
        writers[f].write_table(t)
for w in writers.values():
    w.close()
