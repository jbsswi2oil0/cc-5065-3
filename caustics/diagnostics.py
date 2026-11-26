"""Post-render anomaly checks for the focused spot and the estimator.

The web workbench must show, after every render, whether the photon estimator
actually produced a focused caustic and whether the density estimate shows
signs of failure (undersampling, budget exhaustion, energy-bookkeeping drift,
thin-wall leakage).  Keeping these checks in one place lets the UI, the CLI
and the tests agree on what "focused spot anomaly" and "estimation-error
anomaly" mean.

Every issue is a dict:
    {"code": str, "severity": "warn"|"bad", "area": "spot"|"estimate",
     "message": str, "value": float|None}
"""
from .estimator import UNDER_SAMPLE_PHOTONS

# A focused caustic must clearly out-shine the diffuse direct level; below
# this ratio we treat the "spot" as not meaningfully focused.
FOCUS_RATIO_MIN = 1.5
# Bookkeeping is a per-photon final-fate conservation check; drift beyond this
# indicates an estimator/photon accounting bug, not sampling noise.
BOOKKEEPING_TOL = 1.0e-6
# Back-face caustic energy above this is treated as leakage through the wall.
LEAK_TOL = 1.0e-9
# Fraction of pixels the refiner labelled undersampled that we consider a
# user-actionable estimation problem.
UNDERSAMPLED_WARN = 0.35
UNDERSAMPLED_BAD = 0.60


def _direct_peak(view):
    peak = 0.0
    for row in view["direct"]:
        for px in row:
            peak = max(peak, sum(px) / 3.0)
    return peak


def _caustic_peak(view):
    peak = 0.0
    for row in view["caustic"]:
        for px in row:
            peak = max(peak, sum(px) / 3.0)
    return peak


def diagnose(front_view, back_view, summary):
    """Return a list of spot/estimation anomalies for a rendered result."""
    issues = []
    m = front_view["metrics"]
    spot = m["spot"]
    caustic_peak = _caustic_peak(front_view)
    direct_peak = _direct_peak(front_view)

    # --- focused-spot checks -------------------------------------------
    if m["caustic_irradiance"] <= 1e-12 or caustic_peak <= 0.0:
        issues.append({
            "code": "spot_missing", "severity": "bad", "area": "spot",
            "message": "未检测到聚焦光斑（焦散能量为零）",
            "value": m["caustic_irradiance"],
        })
    else:
        ratio = caustic_peak / direct_peak if direct_peak > 0 else 0.0
        if ratio < FOCUS_RATIO_MIN:
            issues.append({
                "code": "spot_weak_focus", "severity": "warn", "area": "spot",
                "message": (f"聚焦不足：峰值/直接光={ratio:.2f}"
                            f" < {FOCUS_RATIO_MIN:.1f}"),
                "value": ratio,
            })
        cx, cz = spot["centroid"]
        hx, hz = front_view["view_hx"], front_view["view_hz"]
        if abs(cx) > hx or abs(cz) > hz:
            issues.append({
                "code": "spot_off_view", "severity": "warn", "area": "spot",
                "message": f"光斑质心 ({cx:+.2f},{cz:+.2f}) 已移出取样窗口",
                "value": max(abs(cx), abs(cz)),
            })

    # --- estimation-error checks ---------------------------------------
    residual = summary["bookkeeping_residual"]
    if abs(residual) > BOOKKEEPING_TOL:
        issues.append({
            "code": "bookkeeping_drift", "severity": "bad", "area": "estimate",
            "message": f"能量记账残差 {residual:.1e} 超限",
            "value": residual,
        })

    back_leak = back_view["metrics"]["caustic_irradiance"]
    if back_leak > LEAK_TOL:
        issues.append({
            "code": "back_leak", "severity": "bad", "area": "estimate",
            "message": f"墙背面焦散泄漏 {back_leak:.2e}",
            "value": back_leak,
        })

    under = front_view["refine"]["fraction_undersampled"]
    if under >= UNDERSAMPLED_BAD:
        issues.append({
            "code": "undersampled_high", "severity": "bad", "area": "estimate",
            "message": f"{under * 100:.1f}% 像素欠采样，估计误差偏大",
            "value": under,
        })
    elif under >= UNDERSAMPLED_WARN:
        issues.append({
            "code": "undersampled_warn", "severity": "warn", "area": "estimate",
            "message": f"{under * 100:.1f}% 像素欠采样，建议提高预算或放宽半径",
            "value": under,
        })

    budget_exhausted = sum(
        1 for row in front_view["budget_exhausted"] for flag in row if flag)
    total_px = front_view["grid_n"] ** 2
    frac_exhausted = budget_exhausted / total_px
    if frac_exhausted >= 0.05:
        issues.append({
            "code": "budget_exhausted", "severity": "warn", "area": "estimate",
            "message": (f"{frac_exhausted * 100:.1f}% 像素预算耗尽"
                        f"（n_eff<{UNDER_SAMPLE_PHOTONS}）"),
            "value": frac_exhausted,
        })

    # Zero-compatible photons with a lit neighbourhood is a refinement gap,
    # not a truly dark region: flag it as an estimation problem.
    dark = front_view["refine"]["fraction_dark"]
    if dark >= 0.5 and m["total_irradiance"] > 1e-9:
        issues.append({
            "code": "sparse_dark", "severity": "warn", "area": "estimate",
            "message": f"{dark * 100:.1f}% 像素被判为真暗，但与总能量不符",
            "value": dark,
        })

    return issues


def summarize_issues(issues):
    """Compact aggregate for status lines/UI."""
    worst = "ok"
    for it in issues:
        if it["severity"] == "bad":
            worst = "bad"
            break
        if it["severity"] == "warn":
            worst = "warn"
    return {
        "status": worst,
        "count": len(issues),
        "by_area": {
            "spot": sum(1 for i in issues if i["area"] == "spot"),
            "estimate": sum(1 for i in issues if i["area"] == "estimate"),
        },
        "issues": issues,
    }
