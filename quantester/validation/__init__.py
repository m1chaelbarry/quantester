"""Validation package: truncation diagnostics, CPCV/PBO, and research gates."""

from .cpcv import CombinatorialPurgedKFold, PurgedKFold
from .gates import (
    FAIL,
    NOT_APPLICABLE,
    NOT_VALIDATED,
    PASS,
    VALIDATED,
    WARN,
    GateResult,
    ValidationReport,
    build_validation_report,
    evaluate_gates,
    run_cost_stress,
)
from .pbo import pbo_cscv
from .truncation import TruncationResult, run_truncation_test

__all__ = [
    "CombinatorialPurgedKFold",
    "PurgedKFold",
    "pbo_cscv",
    "TruncationResult",
    "run_truncation_test",
    "PASS",
    "WARN",
    "FAIL",
    "NOT_APPLICABLE",
    "VALIDATED",
    "NOT_VALIDATED",
    "GateResult",
    "ValidationReport",
    "evaluate_gates",
    "build_validation_report",
    "run_cost_stress",
]
