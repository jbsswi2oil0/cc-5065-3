#!/usr/bin/env python3
"""Command-line experiment: print bias/normalization, smoothing scale and
energy bookkeeping across photon budgets, plus pane-movement leakage checks.

Usage:
    python3 run_experiment_cli.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caustics.experiments import convergence, run_experiment


def main():
    print("== 固定半径：面积归一化检查（能量应与 N 无关） ==")
    fixed = convergence([3000, 12000, 48000], radius=0.2, max_radius=0.2,
                        adaptive=False)
    for r in fixed:
        print(f"  N={r['n_photons']:6d}  caustic_E={r['front_caustic_energy']:.5f}"
              f"  ratio={r['caustic_energy_ratio']:.4f}"
              f"  peak={r['front_peak']:.4f}  back={r['caustic_back']}")

    print("\n== 自适应半径：清晰度/噪声权衡与平滑尺度 ==")
    adaptive = convergence([3000, 12000, 48000], radius=0.1, max_radius=0.5)
    for r in adaptive:
        print(f"  N={r['n_photons']:6d}  caustic_E={r['front_caustic_energy']:.5f}"
              f"  footprint={r['smoothness_scale']:.4f}"
              f"  undersampled={r['undersampled_fraction']:.3f}"
              f"  residual={r['bookkeeping_residual']:.1e}")

    print("\n== 半径缩小：峰值升高（更锐利）但欠采样上升，能量不变 ==")
    for rad in (0.06, 0.12, 0.25):
        r = run_experiment(n_photons=12000, radius=rad, max_radius=rad,
                           adaptive=False)
        f = r["views"]["wall.front"]
        print(f"  R={rad:.2f}  peak={f['metrics']['spot']['peak']:.4f}"
              f"  caustic_E={f['metrics']['caustic_irradiance']:.4f}"
              f"  undersampled={f['refine']['fraction_undersampled']:.3f}")

    print("\n== 拖动窗棂：光斑位移与背面泄漏（正反两面取样） ==")
    for px in (-1.0, 0.0, 1.0):
        r = run_experiment(pane_x=px, n_photons=12000, radius=0.15,
                           max_radius=0.5, seed=7)
        front = r["views"]["wall.front"]
        back = r["views"]["wall.back"]
        cx, cz = front["metrics"]["spot"]["centroid"]
        print(f"  pane_x={px:+.1f}  spot=({cx:+.3f},{cz:+.3f})"
              f"  front_caustic={front['photon_count']:5d}"
              f"  back_caustic_E={back['metrics']['caustic_irradiance']:.1e}"
              f"  back_photons={back['photon_count']}")

    print("\n== 能量守恒 ==")
    r = run_experiment(n_photons=20000, radius=0.15, max_radius=0.5)
    s = r["tracer_summary"]
    print(f"  emitted={sum(s['emitted_energy']):.4f}"
          f"  absorbed={sum(s['absorbed_energy']):.4f}"
          f"  escaped={sum(s['escaped_energy']):.4f}"
          f"  alive={sum(s['alive_energy']):.4f}")
    print(f"  bookkeeping_residual={s['bookkeeping_residual']:.2e}")

    print("\n== 邻域索引迁移与估计诊断 ==")
    idx = s["index"]
    print(f"  index={idx['backend']}  query_mode={idx['query_mode']}"
          f"  version={idx['version']}  migration={idx['migration']}")
    print(f"  boundary_rule={idx['boundary_rule']}")
    diag = r["diagnostics"]
    print(f"  diagnostics: status={diag['status']}  count={diag['count']}"
          f"  (spot={diag['by_area']['spot']},"
          f" estimate={diag['by_area']['estimate']})")
    for it in diag["issues"]:
        print(f"    [{it['severity']}/{it['area']}] {it['code']}: {it['message']}")


if __name__ == "__main__":
    main()
