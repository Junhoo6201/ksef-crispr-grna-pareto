#!/usr/bin/env python3
"""
Phase 3 보조 — 파레토 전선 크기의 귀무분포와 잡음 교란, q-스캔
(데이터 감사 3차 대응으로 다른 세션이 작성. 본 세션에서 독립 재실행하여 검증.)
출력: out/phase3_front_null.json
"""
import json, pathlib, numpy as np
from scipy.stats import spearmanr, rankdata

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEED, REPS_NULL, REPS_JITTER = 20260906, 2000, 1000

def front_size(pts):
    o = np.lexsort((-pts[:, 1], -pts[:, 0]))
    y = pts[o, 1]
    run = np.maximum.accumulate(np.concatenate([[-np.inf], y[:-1]]))
    return int((y > run).sum())

def main():
    P = np.load(ROOT / "out/phase3_points.npz", allow_pickle=True)
    pred, meas = P["pred"].astype(np.float64), P["meas"].astype(np.float64)
    N = len(pred)
    out = {"n": N, "seed": SEED, "reps_null": REPS_NULL, "reps_jitter": REPS_JITTER}
    rk = lambda p: np.column_stack([rankdata(p[:, 0]), rankdata(p[:, 1])])
    out["observed"] = {
        "predicted": front_size(pred), "measured": front_size(meas),
        "predicted_rank": front_size(rk(pred)), "measured_rank": front_size(rk(meas)),
    }
    out["ties"] = {}
    for nm, p in [("predicted", pred), ("measured", meas)]:
        t = {}
        for ax, lbl in [(0, "Y1"), (1, "Y2")]:
            _, c = np.unique(p[:, ax], return_counts=True)
            t[lbl] = {"tied_groups": int((c > 1).sum()), "tied_points": int(c[c > 1].sum()),
                      "max_tie": int(c.max())}
        out["ties"][nm] = t
    rng = np.random.default_rng(SEED)
    out["null"] = {}
    for nm, p in [("predicted", pred), ("measured", meas)]:
        rho_s = float(spearmanr(p[:, 0], p[:, 1])[0])
        r = 2 * np.sin(np.pi * rho_s / 6)
        L = np.linalg.cholesky([[1, r], [r, 1]])
        s = np.array([front_size(rng.standard_normal((N, 2)) @ L.T) for _ in range(REPS_NULL)])
        obs = out["observed"][nm]
        out["null"][nm] = {
            "spearman": rho_s, "copula_r": float(r),
            "mean": float(s.mean()), "sd": float(s.std(ddof=1)),
            "se_of_mean": float(s.std(ddof=1) / np.sqrt(REPS_NULL)),
            "observed": obs, "z": float((obs - s.mean()) / s.std(ddof=1)),
        }
    out["independent_theory_Hn"] = float(np.log(N) + 0.5772156649)
    sd1 = float((meas[:, 0] - pred[:, 0]).std(ddof=1))
    sd2 = float((meas[:, 1] - pred[:, 1]).std(ddof=1))
    out["residual_sd"] = {"Y1": sd1, "Y2": sd2}
    rng = np.random.default_rng(SEED)
    out["jitter"] = {}
    for k in (0.25, 0.50, 1.00):
        s = np.array([front_size(pred + rng.standard_normal((N, 2)) * [sd1 * k, sd2 * k])
                      for _ in range(REPS_JITTER)])
        out["jitter"][f"{k:.2f}"] = {"mean": float(s.mean()), "sd": float(s.std(ddof=1))}
    out["qscan"] = []
    for q in [0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]:
        row = {"q": q, "expected": N * q * q}
        for nm, p in [("measured", meas), ("predicted", pred)]:
            t1, t2 = np.quantile(p[:, 0], 1 - q), np.quantile(p[:, 1], 1 - q)
            row[nm] = int(((p[:, 0] >= t1) & (p[:, 1] >= t2)).sum())
            row[nm + "_ratio"] = row[nm] / row["expected"]
        out["qscan"].append(row)
    json.dump(out, open(ROOT / "out/phase3_front_null.json", "w"), indent=2, ensure_ascii=False)
    print(f"[front-null] n={N} seed={SEED} null_reps={REPS_NULL} jitter_reps={REPS_JITTER}")
    print(f"  관측 전선   예측 {out['observed']['predicted']} / 측정 {out['observed']['measured']}"
          f"   (순위변환 후 {out['observed']['predicted_rank']} / {out['observed']['measured_rank']})")
    for nm in ("predicted", "measured"):
        d = out["null"][nm]
        print(f"  귀무 {nm:<10} {d['mean']:.4f} ± {d['sd']:.4f} (SE {d['se_of_mean']:.4f})"
              f"  → 관측 {d['observed']} = {d['z']:+.4f}σ")
    print(f"  독립 이론값 H_n = {out['independent_theory_Hn']:.4f}")
    print(f"  잔차 sd  Y1 {sd1:.6f}  Y2 {sd2:.6f}")
    for k, v in out["jitter"].items():
        print(f"  교란 ×{k}: {v['mean']:.4f} ± {v['sd']:.4f}")

if __name__ == "__main__":
    main()
