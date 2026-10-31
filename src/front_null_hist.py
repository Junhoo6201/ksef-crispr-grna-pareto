#!/usr/bin/env python3
"""
그림 4용 히스토그램 자료 — 평가셋 전선 크기 귀무분포·잡음 교란 분포의 '도수'
================================================================
out/phase3_front_null.json 은 평균과 sd 만 담고 있어 히스토그램을 그릴 수 없다.
phase3_front_null.py 와 **같은 방법·같은 시드**로 다시 돌려 도수를 뽑고,
평균·sd 가 기존 JSON 과 일치하는지 검산한 뒤에만 저장한다.
출력: out/front_null_hist.json
"""
import json, pathlib, numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEED, REPS_NULL, REPS_JITTER = 20260906, 2000, 1000
NB = 48                                   # 도수 배열 길이 (전선 크기 0..47)


def front_size(pts):
    o = np.lexsort((-pts[:, 1], -pts[:, 0]))
    y = pts[o, 1]
    run = np.maximum.accumulate(np.concatenate([[-np.inf], y[:-1]]))
    return int((y > run).sum())


def main():
    REF = json.load(open(ROOT / "out/phase3_front_null.json", encoding="utf-8"))
    P = np.load(ROOT / "out/phase3_points.npz", allow_pickle=True)
    pred, meas = P["pred"].astype(np.float64), P["meas"].astype(np.float64)
    N = len(pred)
    out = {"n": N, "seed": SEED, "reps_null": REPS_NULL, "reps_jitter": REPS_JITTER}

    rng = np.random.default_rng(SEED)
    out["null_counts"] = {}
    bad = []
    for nm, p in [("predicted", pred), ("measured", meas)]:
        rho_s = float(spearmanr(p[:, 0], p[:, 1])[0])
        r = 2 * np.sin(np.pi * rho_s / 6)
        L = np.linalg.cholesky([[1, r], [r, 1]])
        s = np.array([front_size(rng.standard_normal((N, 2)) @ L.T) for _ in range(REPS_NULL)])
        out["null_counts"][nm] = np.bincount(s, minlength=NB)[:NB].tolist()
        for key, got in (("mean", s.mean()), ("sd", s.std(ddof=1))):
            ref = REF["null"][nm][key]
            if abs(got - ref) > 1e-9:
                bad.append(f"{nm}.{key}: 재실행 {got!r} vs 기록 {ref!r}")

    sd1, sd2 = REF["residual_sd"]["Y1"], REF["residual_sd"]["Y2"]
    rng = np.random.default_rng(SEED)
    out["jitter_counts"] = {}
    for k in (0.25, 0.50, 1.00):
        s = np.array([front_size(pred + rng.standard_normal((N, 2)) * [sd1 * k, sd2 * k])
                      for _ in range(REPS_JITTER)])
        out["jitter_counts"][f"{k:.2f}"] = np.bincount(s, minlength=NB)[:NB].tolist()
        for key, got in (("mean", s.mean()), ("sd", s.std(ddof=1))):
            ref = REF["jitter"][f"{k:.2f}"][key]
            if abs(got - ref) > 1e-9:
                bad.append(f"jitter{k:.2f}.{key}: 재실행 {got!r} vs 기록 {ref!r}")

    if bad:
        raise SystemExit("[중단] 기존 JSON 과 불일치:\n  " + "\n  ".join(bad))
    json.dump(out, open(ROOT / "out/front_null_hist.json", "w"), ensure_ascii=False, indent=1)
    print("[front-null-hist] 평균·sd 가 out/phase3_front_null.json 과 비트 단위로 일치 — 저장")
    for nm in ("predicted", "measured"):
        c = np.array(out["null_counts"][nm])
        print(f"  귀무 {nm:<10} 도수 합 {c.sum()}  최빈 {c.argmax()}  최대 {np.nonzero(c)[0].max()}")
    c = np.array(out["jitter_counts"]["1.00"])
    print(f"  교란 ×1.00   도수 합 {c.sum()}  최빈 {c.argmax()}  범위 "
          f"{np.nonzero(c)[0].min()}–{np.nonzero(c)[0].max()}")


if __name__ == "__main__":
    main()
