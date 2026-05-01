"""
Tests for tbot.data.loader — load_candles() and candles_to_dict_list().
All tests use synthetic parquet files written to tmp_path; no real data needed.
"""

import pandas as pd
import pytest

from tbot.data.loader import REQUIRED_COLUMNS, candles_to_dict_list, load_candles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parquet(tmp_path, rows: list[dict], filename: str = "test.parquet") -> str:
    """Write a list of dicts to a parquet file and return its path."""
    df = pd.DataFrame(rows)
    path = tmp_path / filename
    df.to_parquet(path, index=False)
    return str(path)


def _minimal_row(i: int, ts_offset_minutes: int = 0) -> dict:
    """One valid OHLCV row with timestamp offset by i*5 + ts_offset_minutes minutes from epoch."""
    base = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    ts = base + pd.Timedelta(minutes=i * 5 + ts_offset_minutes)
    return {
        "timestamp": ts,
        "open":   float(1800 + i),
        "high":   float(1802 + i),
        "low":    float(1799 + i),
        "close":  float(1801 + i),
        "volume": i + 1,
    }


def _make_good_parquet(tmp_path, n: int = 10) -> str:
    rows = [_minimal_row(i) for i in range(n)]
    return _make_parquet(tmp_path, rows)


# ---------------------------------------------------------------------------
# load_candles — happy path
# ---------------------------------------------------------------------------

class TestLoadCandlesHappyPath:
    def test_returns_dataframe(self, tmp_path):
        path = _make_good_parquet(tmp_path)
        df = load_candles(path)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self, tmp_path):
        path = _make_good_parquet(tmp_path)
        df = load_candles(path)
        assert REQUIRED_COLUMNS.issubset(set(df.columns))

    def test_row_count(self, tmp_path):
        path = _make_good_parquet(tmp_path, n=20)
        df = load_candles(path)
        assert len(df) == 20

    def test_timestamp_is_utc_datetime(self, tmp_path):
        path = _make_good_parquet(tmp_path)
        df = load_candles(path)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert hasattr(df["timestamp"].dtype, "tz") and str(df["timestamp"].dtype.tz) == "UTC"

    def test_ohlc_are_float64(self, tmp_path):
        path = _make_good_parquet(tmp_path)
        df = load_candles(path)
        for col in ("open", "high", "low", "close"):
            assert df[col].dtype == "float64", f"{col} should be float64"

    def test_volume_is_int64(self, tmp_path):
        path = _make_good_parquet(tmp_path)
        df = load_candles(path)
        assert df["volume"].dtype == "int64"

    def test_timestamps_monotonically_increasing(self, tmp_path):
        path = _make_good_parquet(tmp_path)
        df = load_candles(path)
        assert df["timestamp"].is_monotonic_increasing

    def test_index_reset(self, tmp_path):
        path = _make_good_parquet(tmp_path, n=5)
        df = load_candles(path)
        assert list(df.index) == list(range(5))


# ---------------------------------------------------------------------------
# load_candles — deduplication
# ---------------------------------------------------------------------------

class TestLoadCandlesDeduplication:
    def test_duplicate_timestamps_removed(self, tmp_path):
        rows = [_minimal_row(0), _minimal_row(0), _minimal_row(1)]  # row 0 duplicated
        path = _make_parquet(tmp_path, rows)
        df = load_candles(path)
        assert len(df) == 2

    def test_no_duplicate_timestamps_in_output(self, tmp_path):
        rows = [_minimal_row(i // 2) for i in range(6)]  # 0,0,1,1,2,2
        path = _make_parquet(tmp_path, rows)
        df = load_candles(path)
        assert df["timestamp"].duplicated().sum() == 0


# ---------------------------------------------------------------------------
# load_candles — date slicing
# ---------------------------------------------------------------------------

class TestLoadCandlesDateSlice:
    def _make_multi_day(self, tmp_path) -> str:
        rows = []
        for day_offset in range(3):  # Jan 1, 2, 3
            for i in range(5):
                base = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
                ts = base + pd.Timedelta(days=day_offset, minutes=i * 5)
                rows.append({
                    "timestamp": ts,
                    "open": 1800.0, "high": 1802.0, "low": 1799.0,
                    "close": 1801.0, "volume": 1,
                })
        return _make_parquet(tmp_path, rows)

    def test_start_filter(self, tmp_path):
        path = self._make_multi_day(tmp_path)
        df = load_candles(path, start="2024-01-02")
        assert df["timestamp"].min() >= pd.Timestamp("2024-01-02", tz="UTC")

    def test_end_filter(self, tmp_path):
        path = self._make_multi_day(tmp_path)
        df = load_candles(path, end="2024-01-01")
        assert df["timestamp"].max() <= pd.Timestamp("2024-01-01", tz="UTC")

    def test_start_and_end_filter(self, tmp_path):
        path = self._make_multi_day(tmp_path)
        df = load_candles(path, start="2024-01-02", end="2024-01-02")
        assert df["timestamp"].min() >= pd.Timestamp("2024-01-02", tz="UTC")
        assert df["timestamp"].max() <= pd.Timestamp("2024-01-02T23:59:59", tz="UTC")

    def test_no_filter_returns_all(self, tmp_path):
        path = self._make_multi_day(tmp_path)
        df = load_candles(path)
        assert len(df) == 15  # 3 days × 5 rows


# ---------------------------------------------------------------------------
# load_candles — error cases
# ---------------------------------------------------------------------------

class TestLoadCandlesErrors:
    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_candles(str(tmp_path / "nonexistent.parquet"))

    def test_missing_column_raises(self, tmp_path):
        rows = [{"timestamp": pd.Timestamp("2024-01-01", tz="UTC"), "open": 1.0}]
        path = _make_parquet(tmp_path, rows)
        with pytest.raises(ValueError, match="missing columns"):
            load_candles(path)

    def test_missing_multiple_columns_raises(self, tmp_path):
        rows = [{"open": 1.0, "high": 2.0}]  # no timestamp, low, close, volume
        path = _make_parquet(tmp_path, rows)
        with pytest.raises(ValueError, match="missing columns"):
            load_candles(path)


# ---------------------------------------------------------------------------
# candles_to_dict_list
# ---------------------------------------------------------------------------

class TestCandlesToDictList:
    def test_returns_list_of_dicts(self, tmp_path):
        path = _make_good_parquet(tmp_path, n=3)
        df = load_candles(path)
        result = candles_to_dict_list(df)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_length_matches_dataframe(self, tmp_path):
        path = _make_good_parquet(tmp_path, n=7)
        df = load_candles(path)
        result = candles_to_dict_list(df)
        assert len(result) == 7

    def test_required_keys_present(self, tmp_path):
        path = _make_good_parquet(tmp_path, n=2)
        df = load_candles(path)
        result = candles_to_dict_list(df)
        expected_keys = {"index", "timestamp", "open", "high", "low", "close", "volume", "is_green", "is_red"}
        for row in result:
            assert expected_keys.issubset(set(row.keys())), f"Missing keys: {expected_keys - set(row.keys())}"

    def test_is_green_correct(self, tmp_path):
        rows = [
            {**_minimal_row(0), "open": 1800.0, "close": 1801.0},  # green
            {**_minimal_row(1), "open": 1801.0, "close": 1800.0},  # red
            {**_minimal_row(2), "open": 1800.0, "close": 1800.0},  # doji
        ]
        path = _make_parquet(tmp_path, rows)
        df = load_candles(path)
        result = candles_to_dict_list(df)
        assert result[0]["is_green"] is True
        assert result[1]["is_green"] is False
        assert result[2]["is_green"] is False

    def test_is_red_correct(self, tmp_path):
        rows = [
            {**_minimal_row(0), "open": 1800.0, "close": 1801.0},  # green
            {**_minimal_row(1), "open": 1801.0, "close": 1800.0},  # red
            {**_minimal_row(2), "open": 1800.0, "close": 1800.0},  # doji
        ]
        path = _make_parquet(tmp_path, rows)
        df = load_candles(path)
        result = candles_to_dict_list(df)
        assert result[0]["is_red"] is False
        assert result[1]["is_red"] is True
        assert result[2]["is_red"] is False

    def test_doji_neither_green_nor_red(self, tmp_path):
        rows = [{**_minimal_row(0), "open": 1800.0, "close": 1800.0}]
        path = _make_parquet(tmp_path, rows)
        df = load_candles(path)
        result = candles_to_dict_list(df)
        assert result[0]["is_green"] is False
        assert result[0]["is_red"] is False

    def test_index_matches_dataframe_index(self, tmp_path):
        path = _make_good_parquet(tmp_path, n=5)
        df = load_candles(path)
        result = candles_to_dict_list(df)
        for i, row in enumerate(result):
            assert row["index"] == i
