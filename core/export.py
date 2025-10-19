"""エクスポートユーティリティ。"""

from __future__ import annotations

import io
import zipfile
from typing import Dict

import pandas as pd


def to_zip(tables: Dict[str, pd.DataFrame]) -> bytes:
    """複数のデータフレームを ZIP (CSV) にまとめる。"""

    buff = io.BytesIO()
    with zipfile.ZipFile(buff, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False, encoding="utf-8-sig"))
    buff.seek(0)
    return buff.read()
