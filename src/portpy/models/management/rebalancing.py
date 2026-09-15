"""
Post-construction portfolio management: rebalancing trade lists, drift/limit
monitoring, and before/after or portfolio-vs-portfolio comparisons.

PortPy's Portfolio itself assumes frictionless, always-rebalanced-to-target
trading (see Portfolio.returns' docstring) - this module is where the
turn-that-assumption-off tools live, for anyone who wants to model the actual
trade list, drift, and cost of getting from one set of weights to another.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from portpy.explain import Explanation, register
from portpy.models.base import ModelResult

__all__ = ["rebalance", "monitor", "compare"]

_REBALANCE_METHODS = ("threshold", "calendar", "full")


def _weights_of(x: pd.Series | ModelResult) -> pd.Series:
    return x.weights if isinstance(x, ModelResult) else x


def rebalance(
    target_weights: pd.Series | ModelResult,
    current_weights: pd.Series,
    method: str = "threshold",
    threshold: float = 0.05,
    cost_bps: float = 0.0,
) -> ModelResult:
    """
    Build a trade list from `current_weights` to `target_weights`.

    Args:
        target_weights: Desired allocation - a plain weight `Series`, or a
            `ModelResult` from `.models.optimize`/`.build` (its `.weights` is used).
        current_weights: The book's actual current weights.
        method: "full"/"calendar" trade all the way to `target_weights`
            regardless of drift size (both produce the same trade list here -
            PortPy's `rebalance` is a stateless, point-in-time trade-list
            generator, not a scheduler, so there's no numerical difference
            between "the calendar says trade now" and "just trade fully";
            a real backtest/strategy engine is where that distinction would
            actually bite, by calling this on a schedule). "threshold" only
            trades assets whose `|current - target|` exceeds `threshold`,
            leaving the rest untouched.
        threshold: Drift threshold for `method="threshold"`, as an absolute weight difference.
        cost_bps: Flat round-trip cost per unit of turnover, in basis points -
            see `metrics.costs.net_of_costs_returns` for the same convention.

    Returns:
        A `ModelResult` named "rebalance". `.weights` is the resulting
        (post-trade) allocation, renormalized to sum to 1. `.diagnostics` has
        `"trades"` (the signed per-asset trade, final - current), `"turnover"`
        (one-way, matching `metrics.costs.turnover_from_weights`'s convention),
        `"estimated_cost"`, `"n_traded"`, `"method"`, and `"threshold"`.

    Raises:
        ValueError: If `method` is unrecognized.
    """
    if method not in _REBALANCE_METHODS:
        raise ValueError(f"method must be one of {_REBALANCE_METHODS}, got {method!r}.")

    target_w = _weights_of(target_weights)
    names = sorted(set(target_w.index) | set(current_weights.index))
    target_w = target_w.reindex(names).fillna(0.0)
    current_w = current_weights.reindex(names).fillna(0.0)

    if method == "threshold":
        drift = (target_w - current_w).abs()
        trade_mask = drift > threshold
        final = current_w.where(~trade_mask, target_w)
    else:
        final = target_w.copy()

    total = final.sum()
    if abs(total) > 1e-12:
        final = final / total
    final = final.rename("weight")

    trades = (final - current_w).rename("trade")
    turnover = float(trades.abs().sum()) / 2.0
    estimated_cost = turnover * (cost_bps / 10_000.0)

    diagnostics = {
        "method": method,
        "threshold": threshold if method == "threshold" else None,
        "turnover": turnover,
        "estimated_cost": estimated_cost,
        "cost_bps": cost_bps,
        "n_traded": int((trades.abs() > 1e-12).sum()),
        "trades": trades,
    }
    return ModelResult(
        name="rebalance", weights=final, diagnostics=diagnostics, meta={"target": target_w, "current": current_w}
    )


def monitor(current_weights: pd.Series, limits: dict[str, Any]) -> dict[str, Any]:
    """
    Check current weights against a set of drift/exposure/concentration limits.

    Args:
        current_weights: The book's current weights.
        limits: Any of:
            - `"max_weight"` / `"min_weight"`: per-asset bounds, checked against every asset.
            - `"max_group"`: `{group_name: cap}`, checked against `limits["groups"]`
              (`{group_name: [asset_names]}`, required if `"max_group"` is given).
            - `"max_gross"`: a cap on `sum(abs(current_weights))`.

    Returns:
        `{"ok": bool, "n_breaches": int, "breaches": [...], "checked": [...]}` -
        each breach is `{"type", ..., "value", "limit"}`, with the middle key
        being `"asset"` or `"group"` depending on `"type"`.
    """
    breaches: list[dict[str, Any]] = []
    w = current_weights

    if "max_weight" in limits:
        for asset, value in w[w > limits["max_weight"]].items():
            breaches.append({"type": "max_weight", "asset": asset, "value": float(value), "limit": limits["max_weight"]})

    if "min_weight" in limits:
        for asset, value in w[w < limits["min_weight"]].items():
            breaches.append({"type": "min_weight", "asset": asset, "value": float(value), "limit": limits["min_weight"]})

    if "max_group" in limits:
        groups: dict[str, list[str]] = limits.get("groups", {})
        for group_name, cap in limits["max_group"].items():
            members = groups.get(group_name, [])
            exposure = float(w.reindex(members).fillna(0.0).sum())
            if exposure > cap:
                breaches.append({"type": "max_group", "group": group_name, "value": exposure, "limit": cap})

    if "max_gross" in limits:
        gross = float(w.abs().sum())
        if gross > limits["max_gross"]:
            breaches.append({"type": "max_gross", "value": gross, "limit": limits["max_gross"]})

    return {"ok": len(breaches) == 0, "n_breaches": len(breaches), "breaches": breaches, "checked": list(w.index)}


def compare(a: Any, b: Any) -> ModelResult:
    """
    Compare two portfolios, model results, or raw weight vectors.

    Args:
        a: A `Portfolio`, `ModelResult`, or weight `Series` - the "before"/baseline side.
        b: Same options - the "after"/candidate side.

    Returns:
        A `ModelResult` named "compare". `.weights` is `b`'s weights.
        `.diagnostics` always has `"mode"` ("returns" if both `a` and `b` are
        `Portfolio` instances with return history, else "weights_only"),
        `"weight_delta"`, `"turnover"`, `"hhi_before"`/`"hhi_after"` (Herfindahl
        concentration), and `"n_active_before"`/`"n_active_after"`. In "returns"
        mode, also `"metrics_before"`/`"metrics_after"` (each side's
        `tearsheet_summary()`) and `"metric_delta"` (after - before, for every
        metric both sides have).
    """
    from portpy.portfolio import Portfolio  # local import: portfolio.py builds .models, which imports this module

    def weights_of(x: Any) -> pd.Series:
        return x.weights if isinstance(x, (ModelResult, Portfolio)) else x

    w_a = weights_of(a)
    w_b = weights_of(b)
    names = sorted(set(w_a.index) | set(w_b.index))
    w_a = w_a.reindex(names).fillna(0.0)
    w_b = w_b.reindex(names).fillna(0.0)

    weight_delta = (w_b - w_a).rename("delta")
    diagnostics: dict[str, Any] = {
        "mode": "weights_only",
        "weight_delta": weight_delta,
        "turnover": float(weight_delta.abs().sum()) / 2.0,
        "hhi_before": float((w_a**2).sum()),
        "hhi_after": float((w_b**2).sum()),
        "n_active_before": int((w_a.abs() > 1e-8).sum()),
        "n_active_after": int((w_b.abs() > 1e-8).sum()),
    }

    if isinstance(a, Portfolio) and isinstance(b, Portfolio):
        summary_a = {k: float(v) for k, v in a.metrics.tearsheet_summary().items()}
        summary_b = {k: float(v) for k, v in b.metrics.tearsheet_summary().items()}
        common = sorted(set(summary_a) & set(summary_b))
        diagnostics["mode"] = "returns"
        diagnostics["metrics_before"] = pd.Series(summary_a)
        diagnostics["metrics_after"] = pd.Series(summary_b)
        diagnostics["metric_delta"] = pd.Series({k: summary_b[k] - summary_a[k] for k in common}, name="delta")

    return ModelResult(name="compare", weights=w_b.rename("weight"), diagnostics=diagnostics, meta={"a": type(a).__name__, "b": type(b).__name__})


register(
    Explanation(
        name="rebalance",
        category="model",
        summary="Turns a target allocation into an actual trade list from where the book currently sits, optionally only trading assets that have drifted past a threshold.",
        formula="threshold: final_i = target_i if |target_i - current_i| > threshold else current_i, then renormalize to sum to 1 | full/calendar: final = target",
        how_to_read="diagnostics['trades'] is the signed change per asset (positive = buy, negative = sell); diagnostics['turnover'] is the one-way fraction of the book traded, same convention as metrics.costs.turnover_from_weights.",
        good_vs_bad="Lower turnover for a given amount of drift correction is generally better (less cost/tax drag) - that's the whole rationale for method='threshold' over always trading fully back to target.",
        caveats="'calendar' and 'full' produce identical trade lists here - PortPy's rebalance() is a stateless point-in-time function, not a scheduler, so it can't distinguish 'it's the scheduled date' from 'trade all the way now'; that distinction only matters inside an actual backtest loop (Stage 5). The post-threshold renormalization to sum-to-1 means an asset just under the threshold can still see its weight shift slightly even though it wasn't itself traded. estimated_cost uses the same flat, linear-in-turnover model as metrics.costs.net_of_costs_returns - no market impact, no tax treatment (see metrics.costs for what's actually implemented).",
        interpret=lambda r: f"turnover={r.diagnostics['turnover']:.2%}, {r.diagnostics['n_traded']} asset(s) traded, est. cost={r.diagnostics['estimated_cost']:.4%}",
    )
)

register(
    Explanation(
        name="compare",
        category="model",
        summary="Compares two portfolios, optimizer results, or raw weight vectors - weight-level always, and full metric-level whenever both sides are Portfolios with their own return history.",
        formula="weight_delta = b - a; turnover = sum(abs(weight_delta))/2; HHI = sum(w^2); metric_delta = tearsheet_summary(b) - tearsheet_summary(a) [Portfolio vs. Portfolio only]",
        how_to_read="hhi_before/hhi_after are Herfindahl concentration indices (1/N for equal-weight, 1.0 for fully concentrated in one asset) - rising HHI means the candidate is more concentrated, not necessarily worse.",
        good_vs_bad="In 'returns' mode, a positive metric_delta on return/Sharpe-style metrics and a controlled (not excessive) turnover is the usual 'good' outcome for a proposed switch; judge concentration change (HHI) against your own diversification preference, not a universal target.",
        caveats="Falls back to 'weights_only' mode (no metric_delta) whenever either side isn't a full Portfolio (e.g. comparing a live Portfolio against a bare optimizer ModelResult that has no return history) - check diagnostics['mode'] before assuming metric_delta exists. metric_delta only includes metrics present on *both* sides.",
        interpret=lambda r: (
            f"turnover={r.diagnostics['turnover']:.2%}, HHI {r.diagnostics['hhi_before']:.3f}->{r.diagnostics['hhi_after']:.3f}"
            + (f", {len(r.diagnostics['metric_delta'])} metrics compared" if r.diagnostics.get("mode") == "returns" else " (weights-only, no return history on both sides)")
        ),
    )
)

register(
    Explanation(
        name="monitor",
        category="function",
        summary="Checks a portfolio's current weights against a set of hard limits (per-asset bounds, group/sector caps, gross exposure) and reports any breaches.",
        formula="breach if weight_i > max_weight, weight_i < min_weight, sum(weights in group) > max_group[group], or sum(abs(weights)) > max_gross",
        how_to_read="An empty breaches list (ok=True) means every supplied limit is currently satisfied - monitor only checks the limits you actually pass in `limits`, it has no built-in defaults.",
        good_vs_bad="ok=True is the goal; each entry in breaches identifies exactly which limit and by how much, so you can decide whether to rebalance() back toward compliance.",
        caveats="A point-in-time check only - it says nothing about how long a breach has persisted or how it's trending. 'max_group' requires you to supply `limits['groups']` yourself (e.g. built from Portfolio.asset_classes) - PortPy doesn't infer groupings automatically.",
        interpret=lambda d: "OK - no breaches" if d["ok"] else f"{d['n_breaches']} breach(es): " + ", ".join(f"{b['type']}({b.get('asset', b.get('group', ''))})" for b in d["breaches"]),
    )
)
