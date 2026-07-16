import pytest

from backend.ml.features.technical import add_technical_features


EXPECTED_TECHNICAL_COLS = [
    "price_to_sma10", "price_to_sma20",
    "vol_5", "vol_20",
    "rsi14",
    "macd", "macd_signal",
    "bb_width",
    "dow_sin", "dow_cos",
]


def test_all_expected_columns_present(technical_df):
    for col in EXPECTED_TECHNICAL_COLS:
        assert col in technical_df.columns, f"Missing column: {col}"


def test_no_nans_in_output(technical_df):
    assert not technical_df.isnull().any().any()


def test_dow_sin_cos_bounded(technical_df):
    assert (technical_df["dow_sin"].abs() <= 1.0 + 1e-9).all()
    assert (technical_df["dow_cos"].abs() <= 1.0 + 1e-9).all()


def test_bb_width_non_negative(technical_df):
    assert (technical_df["bb_width"] >= 0).all()


def test_output_shorter_than_input(feature_df, technical_df):
    assert len(technical_df) < len(feature_df)


def test_rsi14_bounded(technical_df):
    assert (technical_df["rsi14"] >= -1.0 - 1e-9).all()
    assert (technical_df["rsi14"] <= 1.0 + 1e-9).all()
