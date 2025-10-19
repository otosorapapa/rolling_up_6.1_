"""入出力ユーティリティ。

CSV/Excel の読み込みやエンコーディング判定を行う。
"""

from __future__ import annotations

import io

import chardet
import pandas as pd


def detect_encoding(data: bytes) -> str:
    """バイト列から推定される文字コードを返す。"""
    result = chardet.detect(data)
    return result.get("encoding", "utf-8")


def read_table(file: io.BytesIO, filename: str) -> pd.DataFrame:
    """CSV/Excel ファイルを読み込む。

    Args:
        file: アップロードされたバイト列。
        filename: 拡張子判定用の元ファイル名。

    Returns:
        ``pd.DataFrame``
    """

    data = file.read()
    file.seek(0)
    if filename.lower().endswith(".csv"):
        enc = detect_encoding(data)
        return pd.read_csv(io.BytesIO(data), encoding=enc)
    else:
        return pd.read_excel(file)
