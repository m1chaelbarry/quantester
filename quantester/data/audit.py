"""Dataset-quality audit for point-in-time OHLCV research datasets.

Statuses are ``PASS``, ``WARN``, or ``FAIL``. Warnings are never silently
promoted to passes. ``DataAuditReport.passed`` / ``data_validated`` require
a clean PASS; ``data_valid`` allows WARN (no FAIL). Production workflows
must require ``data_validated`` (or ``passed``), not merely ``data_valid``.
Corporate-action, survivorship, and universe-membership checks are
documentation/assumption gates when the dataset does not carry explicit
metadata — they surface as WARN until the caller documents them.

Verification status: not covered by the notebook — implemented as an
engineering data-integrity layer for Quantester's retail research workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .streaming import REQUIRED_COLUMNS

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""

    def __str__(self) -> str:
        suffix = f" — {self.detail}" if self.detail else ""
        return f"[{self.status}] {self.name}{suffix}"


@dataclass
class DataAuditReport:
    symbol: str
    checks: list[CheckResult] = field(default_factory=list)
    assumptions: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        statuses = {c.status for c in self.checks}
        if FAIL in statuses:
            return FAIL
        if WARN in statuses:
            return WARN
        return PASS

    @property
    def passed(self) -> bool:
        """True only on a clean PASS (warnings do not count as passed)."""
        return self.status == PASS

    @property
    def data_valid(self) -> bool:
        """Structural integrity: no FAIL checks (WARN allowed)."""
        return self.status != FAIL

    @property
    def data_valid_with_warnings(self) -> bool:
        return self.status == WARN

    @property
    def data_validated(self) -> bool:
        """Production-ready: all checks PASS including documentation gates."""
        return self.status == PASS

    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == FAIL]

    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == WARN]

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail}
                for c in self.checks
            ],
            "assumptions": dict(self.assumptions),
        }


def _check_timezone(index: pd.DatetimeIndex) -> CheckResult:
    if not isinstance(index, pd.DatetimeIndex):
        return CheckResult("timestamp_timezone_explicit", FAIL, "index is not DatetimeIndex")
    if index.tz is None:
        return CheckResult(
            "timestamp_timezone_explicit",
            FAIL,
            "timestamps are tz-naive; normalize to timezone-aware UTC at ingestion",
        )
    if str(index.tz) not in {"UTC", "utc", "tzutc()"} and getattr(index.tz, "zone", None) != "UTC":
        # Accept any UTC-equivalent tzinfo (pytz/zoneinfo/dateutil).
        try:
            offset = index[0].utcoffset() if len(index) else None
            if offset is not None and offset.total_seconds() == 0:
                return CheckResult("timestamp_timezone_explicit", PASS, "UTC-equivalent tz")
        except (AttributeError, TypeError, ValueError):
            # Non-standard tzinfo without a usable utcoffset — fall through to WARN.
            pass
        return CheckResult(
            "timestamp_timezone_explicit",
            WARN,
            f"timezone is {index.tz}; prefer UTC for the core engine",
        )
    return CheckResult("timestamp_timezone_explicit", PASS, "UTC")


def _check_monotonic(index: pd.DatetimeIndex) -> CheckResult:
    if not index.is_monotonic_increasing:
        return CheckResult("timestamps_monotonic", FAIL, "index is not monotonic increasing")
    return CheckResult("timestamps_monotonic", PASS)


def _check_duplicates(index: pd.DatetimeIndex) -> CheckResult:
    n_dup = int(index.duplicated().sum())
    if n_dup:
        return CheckResult("no_duplicate_timestamps", FAIL, f"{n_dup} duplicate timestamp(s)")
    return CheckResult("no_duplicate_timestamps", PASS)


def _check_ohlc(df: pd.DataFrame) -> list[CheckResult]:
    out = []
    ohlcv = df[list(REQUIRED_COLUMNS)]
    if ohlcv.isna().any().any() or not np.isfinite(ohlcv.to_numpy(dtype=float)).all():
        n = int((~np.isfinite(ohlcv.to_numpy(dtype=float))).sum())
        out.append(
            CheckResult(
                "ohlcv_finite",
                FAIL,
                f"{n} non-finite (NaN/inf) OHLCV value(s)",
            )
        )
    else:
        out.append(CheckResult("ohlcv_finite", PASS))
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    if (h < np.maximum(o, c)).any() or (l > np.minimum(o, c)).any() or (h < l).any():
        n = int(((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)).sum())
        out.append(CheckResult("ohlc_relationships_valid", FAIL, f"{n} invalid OHLC bar(s)"))
    else:
        out.append(CheckResult("ohlc_relationships_valid", PASS))
    if (df[list(REQUIRED_COLUMNS[:4])] <= 0).any().any():
        out.append(CheckResult("prices_positive", FAIL, "non-positive price(s) present"))
    else:
        out.append(CheckResult("prices_positive", PASS))
    if (df["volume"] < 0).any():
        out.append(CheckResult("volume_non_negative", FAIL, "negative volume present"))
    else:
        out.append(CheckResult("volume_non_negative", PASS))
    zero_vol = int((df["volume"] == 0).sum())
    if zero_vol:
        out.append(
            CheckResult(
                "suspicious_zero_volume_bars",
                WARN,
                f"{zero_vol} zero-volume bar(s)",
            )
        )
    else:
        out.append(CheckResult("suspicious_zero_volume_bars", PASS))
    return out


def _check_missing_bars(index: pd.DatetimeIndex, freq: str | None) -> CheckResult:
    if freq is None or len(index) < 2:
        return CheckResult(
            "missing_bars_identified",
            WARN,
            "trading calendar / expected frequency not provided",
        )
    expected = pd.date_range(index[0], index[-1], freq=freq, tz=index.tz)
    missing = expected.difference(index)
    if len(missing):
        return CheckResult(
            "missing_bars_identified",
            WARN,
            f"{len(missing)} gap(s) vs freq={freq}",
        )
    return CheckResult("missing_bars_identified", PASS, f"no gaps vs freq={freq}")


def audit_ohlcv_frame(
    df: pd.DataFrame,
    symbol: str = "",
    *,
    expected_freq: str | None = None,
    adjustment_policy: str | None = None,
    corporate_actions_documented: bool = False,
    delistings_considered: bool = False,
    survivorship_bias_considered: bool = False,
    historical_universe_documented: bool = False,
    trading_calendar_documented: bool = False,
) -> DataAuditReport:
    """Run the reusable dataset-quality checklist on one OHLCV frame."""
    report = DataAuditReport(symbol=symbol or "<unknown>")
    assumptions = {
        "adjustment_policy": adjustment_policy,
        "corporate_actions_documented": corporate_actions_documented,
        "delistings_considered": delistings_considered,
        "survivorship_bias_considered": survivorship_bias_considered,
        "historical_universe_documented": historical_universe_documented,
        "trading_calendar_documented": trading_calendar_documented,
        "expected_freq": expected_freq,
    }
    report.assumptions = assumptions

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report.checks.append(
            CheckResult("required_ohlcv_columns", FAIL, f"missing {missing_cols}")
        )
        return report
    report.checks.append(CheckResult("required_ohlcv_columns", PASS))

    idx = pd.DatetimeIndex(df.index)
    report.checks.append(_check_timezone(idx))
    report.checks.append(_check_monotonic(idx))
    report.checks.append(_check_duplicates(idx))
    report.checks.extend(_check_ohlc(df))
    report.checks.append(_check_missing_bars(idx, expected_freq))

    # Documentation / research-governance gates — WARN until caller documents.
    def _doc_gate(name: str, ok: bool, detail_ok: str, detail_warn: str) -> CheckResult:
        return CheckResult(name, PASS if ok else WARN, detail_ok if ok else detail_warn)

    report.checks.append(
        _doc_gate(
            "adjusted_unadjusted_price_semantics",
            adjustment_policy is not None,
            f"policy={adjustment_policy}",
            "adjustment / unadjusted price semantics not documented",
        )
    )
    report.checks.append(
        _doc_gate(
            "corporate_action_treatment",
            corporate_actions_documented,
            "documented",
            "corporate-action treatment not documented",
        )
    )
    report.checks.append(
        _doc_gate(
            "delistings_considered",
            delistings_considered,
            "documented",
            "delistings not explicitly considered",
        )
    )
    report.checks.append(
        _doc_gate(
            "survivorship_bias_considered",
            survivorship_bias_considered,
            "documented",
            "survivorship bias not explicitly considered",
        )
    )
    report.checks.append(
        _doc_gate(
            "historical_universe_membership",
            historical_universe_documented,
            "documented",
            "historical universe membership not documented",
        )
    )
    report.checks.append(
        _doc_gate(
            "trading_calendar_documented",
            trading_calendar_documented or expected_freq is not None,
            "documented",
            "trading calendar not documented",
        )
    )
    return report


def audit_multi_symbol(
    frames: dict,
    *,
    expected_freq: str | None = None,
    **assumption_kwargs,
) -> dict:
    """Audit each symbol and report cross-asset timestamp alignment."""
    per_symbol = {
        symbol: audit_ohlcv_frame(
            df, symbol, expected_freq=expected_freq, **assumption_kwargs
        )
        for symbol, df in frames.items()
    }
    indexes = [pd.DatetimeIndex(df.index) for df in frames.values()]
    alignment_checks = []
    if indexes:
        tzs = {str(idx.tz) for idx in indexes}
        if len(tzs) > 1:
            alignment_checks.append(
                CheckResult(
                    "cross_asset_timezone_alignment",
                    FAIL,
                    f"mixed timezones {sorted(tzs)}",
                )
            )
        else:
            alignment_checks.append(CheckResult("cross_asset_timezone_alignment", PASS))
        union = indexes[0]
        for idx in indexes[1:]:
            union = union.union(idx)
        intersection = indexes[0]
        for idx in indexes[1:]:
            intersection = intersection.intersection(idx)
        coverage = len(intersection) / max(len(union), 1)
        if coverage < 1.0:
            alignment_checks.append(
                CheckResult(
                    "cross_asset_timestamp_alignment",
                    WARN,
                    f"intersection/union={coverage:.2%}; outer-join + masks required",
                )
            )
        else:
            alignment_checks.append(
                CheckResult("cross_asset_timestamp_alignment", PASS, "identical calendars")
            )
    statuses = {r.status for r in per_symbol.values()} | {c.status for c in alignment_checks}
    overall = FAIL if FAIL in statuses else (WARN if WARN in statuses else PASS)
    return {
        "status": overall,
        "symbols": {s: r.to_dict() for s, r in per_symbol.items()},
        "alignment": [
            {"name": c.name, "status": c.status, "detail": c.detail}
            for c in alignment_checks
        ],
    }


def ensure_utc_index(index) -> pd.DatetimeIndex:
    """Normalize timestamps to timezone-aware UTC.

    - tz-naive indexes are localized as UTC (provider contract: naive means UTC
      for exchange-provided epoch timestamps; exchange-local wall times must be
      localized by the provider before this helper runs).
    - tz-aware indexes are converted to UTC.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx.tz_localize("UTC").as_unit("ns")
    return idx.tz_convert("UTC").as_unit("ns")
