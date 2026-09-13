#!/usr/bin/env python3
"""
Phase 3 보조 — q-스캔의 귀무분포
================================
phase3_front_null.py 의 q-스캔은 관측 수를 '독립 기대값 Nq²' 와 비교한다.
그러나 두 목적은 완전 독립이 아니다(측정 스피어만 +0.0324). 따라서 Nq² 는
기준으로 약간 낮고, 그만큼 높은 q 에서 과잉을, 낮은 q 에서 결핍을 왜곡한다.

여기서는 실제 스피어만을 맞춘 가우시안 copula 를 귀무로 삼아 각 q 에서
동시 상위 개수의 분포를 구하고, 관측값의 z 와 백분위를 낸다.

출력: out/phase3_qscan_null.json
실행: python3 src/phase3_qscan_null.py
"""
import json, pathlib, numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEED, REPS = 20260906, 2000
QS = [0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]


def joint_top(p, q):
    t1, t2 = np.quantile(p[:, 0], 1 - q), np.quantile(p[:, 1], 1 - q)
    return int(((p[:, 0] >= t1) & (p[:, 1] >= t2)).sum())


def main():
    P = np.load(ROOT / "out/phase3_points.npz", allow_pickle=True)
    pred, meas = P["pred"].astype(np.float64), P["meas"].astype(np.float64)
    N = len(meas)
    out = {"n": N, "seed": SEED, "reps": REPS, "q_values": QS,
           "null_model": "실제 스피어만을 맞춘 가우시안 copula", "series": {}}

    for nm, p in [("measured", meas), ("predicted", pred)]:
        rho = float(spearmanr(p[:, 0], p[:, 1])[0])
        r = 2 * np.sin(np.pi * rho / 6)
        L = np.linalg.cholesky([[1, r], [r, 1]])
        rng = np.random.default_rng(SEED)
        sims = {q: [] for q in QS}
        for _ in range(REPS):
            S = rng.standard_normal((N, 2)) @ L.T
            for q in QS:
                sims[q].append(joint_top(S, q))
        rows = []
        for q in QS:
            s = np.array(sims[q]); o = joint_top(p, q)
            sd = s.std(ddof=1)
            rows.append(dict(
                q=q, observed=o,
                independent_Nq2=N * q * q,
                null_mean=float(s.mean()), null_sd=float(sd),
                z=float((o - s.mean()) / sd) if sd > 0 else None,
                percentile=float((s <= o).mean() * 100),
                ratio_vs_Nq2=o / (N * q * q),
                ratio_vs_null=o / float(s.mean()),
            ))
        out["series"][nm] = dict(spearman=rho, copula_r=float(r), rows=rows)

    # 부호가 바뀌는 구간 (z 가 0 을 지나는 인접 q 쌍)
    out["sign_change"] = {}
    for nm in ("measured", "predicted"):
        rr = out["series"][nm]["rows"]
        seg = next(((rr[i]["q"], rr[i + 1]["q"]) for i in range(len(rr) - 1)
                    if rr[i]["z"] >= 0 > rr[i + 1]["z"]), None)
        sig = next((r["q"] for r in rr if r["z"] is not None and r["z"] <= -2), None)
        out["sign_change"][nm] = {"z_crosses_zero_between": seg,
                                  "first_q_with_z_le_minus2": sig}

    json.dump(out, open(ROOT / "out/phase3_qscan_null.json", "w"), indent=2, ensure_ascii=False)
    for nm in ("measured", "predicted"):
        S = out["series"][nm]
        print(f"\n=== {nm}  (스피어만 {S['spearman']:+.4f}) ===")
        print(f"{'q':>5} {'Nq²':>8} {'귀무평균':>9} {'sd':>6} {'관측':>5} {'z':>7} {'백분위':>7}")
        for r in S["rows"]:
            print(f"{r['q']:>5.2f} {r['independent_Nq2']:>8.1f} {r['null_mean']:>9.2f} "
                  f"{r['null_sd']:>6.2f} {r['observed']:>5d} {r['z']:>+7.2f} {r['percentile']:>6.1f}%")
        print(f"  z 부호전환 구간 {out['sign_change'][nm]['z_crosses_zero_between']}"
              f" · z ≤ −2 첫 지점 q = {out['sign_change'][nm]['first_q_with_z_le_minus2']}")
    print("\n[save] out/phase3_qscan_null.json")


if __name__ == "__main__":
    main()
