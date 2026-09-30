from pathlib import Path
import pandas as pd


def read_snapshot(path, columns=None):
    p = Path(path)
    name = p.name.lower()
    if name.endswith('.parquet'):
        return pd.read_parquet(p, columns=columns)
    if name.endswith('.csv') or name.endswith('.csv.gz'):
        df = pd.read_csv(p, low_memory=False)
        if columns is not None:
            df = df[columns]
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        return df
    if name.endswith('.pkl') or name.endswith('.pkl.gz'):
        df = pd.read_pickle(p)
        return df[columns] if columns is not None else df
    raise ValueError(f'Unsupported dataset artifact: {p}')
