import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.metrics import PVMResult, mat, pvm


def test_mat_basic():
    s = pd.Series(np.arange(1, 25))
    result = mat(s, window=12)
    assert result.isna().sum() == 11
    assert result.dropna().iloc[0] == s.iloc[:12].sum()


def test_pvm_decomposition():
    df0 = pd.DataFrame({"unit_price": [100, 200], "qty": [10, 5]}, index=["A", "B"])
    df1 = pd.DataFrame({"unit_price": [110, 190], "qty": [9, 6]}, index=["A", "B"])
    res = pvm(df0, df1)
    assert isinstance(res, PVMResult)
    recon = res.price_effect + res.volume_effect + res.mix_effect
    assert res.actual_diff == pytest.approx(recon, rel=1e-3)
