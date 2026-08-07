"""Validation gate system for research governance.

Each gate returns PASS / WARN / FAIL / NOT_APPLICABLE. A research run may only
claim ``VALIDATED`` when every mandatory gate is PASS (or NOT_APPLICABLE).
Warnings remain visible and never silently become passes.

These gates are diagnostics and governance controls — they do not prove a
strategy is profitable, unbiased, or free of all look-ahead.

Verification status: not covered by the notebook — engineering governance
layer over Quantester's existing statistical tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
NOT_APPLICABLE = "NOT_APPLICABLE"

VALIDATED = "VALIDATED"
NOT_VALIDATED = "NOT_VALIDATED"


@dataclass
class GateResult:
    name: str
    status: str
    detail: str = ""
    mandatory: bool = True
    metrics: dict = field(default_factory=dict)

    def __str__(self) -> str:
        flag = "mandatory" if self.mandatory else "optional"
        suffix = f" — {self.detail}" if self.detail else ""
        return f"[{self.status}] {self.name} ({flag}){suffix}"


@dataclass
class ValidationReport:
    gates: list[GateResult] = field(default_factory=list)
    assumptions: dict = field(default_factory=dict)
    trial_count: int | None = None
    experiment_ids: list[str] = field(default_factory=list)
    code_version: str | None = None
    random_seeds: dict = field(default_factory=dict)
    execution_config: dict = field(default_factory=dict)
    data_config: dict = field(default_factory=dict)
    performance: dict = field(default_factory=dict)
    robustness: dict = field(default_factory=dict)
    untouched_oos: dict | None = None

    @property
    def status(self) -> str:
        for g in self.gates:
            if g.mandatory and g.status == FAIL:
                return NOT_VALIDATED
        if any(g.status == FAIL for g in self.gates):
            return NOT_VALIDATED
        if any(g.status == WARN for g in self.gates):
            return WARN
        if not self.gates:
            return NOT_VALIDATED
        return VALIDATED if all(
            g.status in {PASS, NOT_APPLICABLE} for g in self.gates if g.mandatory
        ) else WARN

    @property
    def validated(self) -> bool:
        return self.status == VALIDATED

    def failures(self) -> list[GateResult]:
        return [g for g in self.gates if g.status == FAIL]

    def warnings(self) -> list[GateResult]:
        return [g for g in self.gates if g.status == WARN]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "validated": self.validated,
            "trial_count": self.trial_count,
            "experiment_ids": list(self.experiment_ids),
            "code_version": self.code_version,
            "random_seeds": dict(self.random_seeds),
            "execution_config": dict(self.execution_config),
            "data_config": dict(self.data_config),
            "assumptions": dict(self.assumptions),
            "performance": dict(self.performance),
            "robustness": dict(self.robustness),
            "untouched_oos": self.untouched_oos,
            "gates": [
                {
                    "name": g.name,
                    "status": g.status,
                    "detail": g.detail,
                    "mandatory": g.mandatory,
                    "metrics": g.metrics,
                }
                for g in self.gates
            ],
        }

    def summary_text(self) -> str:
        lines = [
            f"Validation status: {self.status}",
            f"Trials registered: {self.trial_count}",
            f"Code version: {self.code_version}",
            "",
            "Gates:",
        ]
        for g in self.gates:
            lines.append(f"  {g}")
        if self.assumptions:
            lines.append("")
            lines.append("Assumptions:")
            for k, v in self.assumptions.items():
                lines.append(f"  - {k}: {v}")
        return "\n".join(lines)


def gate_from_bool(
    name: str,
    ok: bool | None,
    *,
    detail: str = "",
    mandatory: bool = True,
    metrics: dict | None = None,
    warn_instead_of_fail: bool = False,
) -> GateResult:
    if ok is None:
        status = NOT_APPLICABLE
    elif ok:
        status = PASS
    elif warn_instead_of_fail:
        status = WARN
    else:
        status = FAIL
    return GateResult(
        name=name,
        status=status,
        detail=detail,
        mandatory=mandatory,
        metrics=metrics or {},
    )


def evaluate_gates(
    *,
    data_audit_status: str | None = None,
    truncation_passed: bool | None = None,
    parity_passed: bool | None = None,
    execution_stress_passed: bool | None = None,
    cpcv_passed: bool | None = None,
    pbo_passed: bool | None = None,
    pbo_value: float | None = None,
    dsr_value: float | None = None,
    dsr_threshold: float = 0.95,
    untouched_oos_passed: bool | None = None,
    monte_carlo_passed: bool | None = None,
    sensitivity_passed: bool | None = None,
    accounting_invariant_passed: bool | None = None,
    execution_assumptions_documented: bool | None = None,
    extra_gates: Iterable[GateResult] = (),
) -> list[GateResult]:
    """Build the recommended gate list from discrete research outcomes."""
    gates: list[GateResult] = []

    if data_audit_status is None:
        gates.append(gate_from_bool("data_audit", None, detail="not run"))
    else:
        status = data_audit_status.upper()
        if status not in {PASS, WARN, FAIL}:
            raise ValueError(f"invalid data_audit_status {data_audit_status!r}")
        gates.append(
            GateResult(
                name="data_audit",
                status=status,
                detail="dataset-quality audit",
                mandatory=True,
            )
        )

    gates.append(
        gate_from_bool(
            "temporal_truncation",
            truncation_passed,
            detail="Chan truncation diagnostic (not a formal look-ahead proof)",
        )
    )
    gates.append(
        gate_from_bool(
            "event_vectorized_parity",
            parity_passed,
            detail="where a vectorized twin exists",
            mandatory=False,
        )
    )
    gates.append(
        gate_from_bool(
            "execution_cost_stress",
            execution_stress_passed,
            detail="BASE/CONSERVATIVE/STRESS viability",
        )
    )
    gates.append(gate_from_bool("cpcv", cpcv_passed, detail="combinatorial purged CV"))
    detail = f"PBO={pbo_value:.4f}" if pbo_value is not None else ""
    gates.append(gate_from_bool("pbo", pbo_passed, detail=detail, metrics={"pbo": pbo_value}))
    if dsr_value is None:
        gates.append(gate_from_bool("dsr", None, detail="DSR not computed"))
    else:
        gates.append(
            gate_from_bool(
                "dsr",
                dsr_value >= dsr_threshold,
                detail=f"DSR={dsr_value:.4f} (threshold {dsr_threshold})",
                metrics={"dsr": dsr_value, "threshold": dsr_threshold},
            )
        )
    gates.append(
        gate_from_bool(
            "untouched_oos",
            untouched_oos_passed,
            detail="holdout must not be reused for parameter selection",
        )
    )
    gates.append(
        gate_from_bool(
            "monte_carlo_robustness",
            monte_carlo_passed,
            detail="diagnostic-aware resampling",
        )
    )
    gates.append(
        gate_from_bool(
            "sensitivity_analysis",
            sensitivity_passed,
            detail="parameter / cost sensitivity",
            mandatory=False,
        )
    )
    gates.append(
        gate_from_bool(
            "portfolio_accounting_invariants",
            accounting_invariant_passed,
            detail="equity = cash + MTM",
        )
    )
    gates.append(
        gate_from_bool(
            "execution_assumptions_documented",
            execution_assumptions_documented,
            detail="spread / slippage / impact / participation",
        )
    )
    gates.extend(list(extra_gates))
    return gates


def build_validation_report(
    gates: list[GateResult] | None = None,
    **report_kwargs: Any,
) -> ValidationReport:
    """Assemble a final validation report; ``VALIDATED`` is blocked by mandatory FAIL."""
    report = ValidationReport(gates=list(gates or []), **{
        k: v for k, v in report_kwargs.items() if k in ValidationReport.__dataclass_fields__
    })
    # Defensive: never allow VALIDATED with a mandatory FAIL.
    if any(g.mandatory and g.status == FAIL for g in report.gates):
        assert not report.validated
    return report


def run_cost_stress(
    run_fn: Callable[[Any], dict],
    scenarios: dict | None = None,
) -> GateResult:
    """Evaluate BASE / CONSERVATIVE / STRESS cost scenarios via a caller hook.

    ``run_fn(cost_model)`` must return a dict with at least ``viable: bool``.
    """
    from ..execution.costs import retail_cost_scenario

    scenarios = scenarios or {
        name: retail_cost_scenario(name) for name in ("BASE", "CONSERVATIVE", "STRESS")
    }
    results = {}
    for name, model in scenarios.items():
        out = run_fn(model)
        results[name] = out
    all_viable = all(bool(v.get("viable", False)) for v in results.values())
    detail = ", ".join(
        f"{k}:{'ok' if v.get('viable') else 'fail'}" for k, v in results.items()
    )
    return GateResult(
        name="execution_cost_stress",
        status=PASS if all_viable else FAIL,
        detail=detail,
        mandatory=True,
        metrics=results,
    )
