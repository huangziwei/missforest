import numpy as np
import pandas as pd
import pytest

from missforest import MissForest  # replace 'your_module' with the actual module name


def test_instantiation():
    mf = MissForest()
    assert isinstance(mf, MissForest)


def test_fit():
    mf = MissForest()
    X = pd.DataFrame({"num": [1, 2, np.nan, 4], "cat": ["A", "B", "C", np.nan]})
    mf.fit(X)
    assert mf.column_order == ["num", "cat"]
    assert mf.num_cols[0] == "num"
    assert mf.cat_cols[0] == "cat"


def test_transform():
    mf = MissForest()
    X = pd.DataFrame({"num": [1, 2, np.nan, 4], "cat": ["A", "B", "C", np.nan]})
    mf.fit(X)
    transformed = mf.transform(X)
    assert len(transformed) == mf.n_imputations
    assert not any([df.isnull().any().any() for df in transformed])


def test_fit_transform():
    mf = MissForest()
    X = pd.DataFrame({"num": [1, 2, np.nan, 4], "cat": ["A", "B", "C", np.nan]})
    transformed = mf.fit_transform(X)
    assert len(transformed) == mf.n_imputations
    assert not any([df.isnull().any().any() for df in transformed])
