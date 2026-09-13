"""
Phase 6 시드 스윕 — 사전등록 부록 (2026-09-12)
=============================================
커밋된 phase6_baseline.py 를 수정하지 않고 함수만 import 하여 호출한다.
시드 베이스 b 에 대해 원본과 동일한 절차를 seed = b+1, b+2, b+3, b+4 로 수행한다.
b = 0 은 커밋된 절차(1,2,3,4)와 정확히 같다.

10% 조건의 부분표본 추출 난수는 원본과 같은 20260830 으로 고정한다 —
모델 시드만 변하게 하여 잡음의 출처를 하나로 묶기 위해서다.

출력: out/phase6_seed_sweep.json  (새 파일)
※ out/phase6_baseline.json 은 건드리지 않는다.
"""
import json, pathlib, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import phase6_baseline as P6

SEED_BASES = [0, 10, 20, 30, 40]                          # 실행 전 고정

if __name__ == "__main__":
    D = dict(np.load(P6.ROOT / "out/dataset.npz", allow_pickle=True))
    raw = pd.read_csv(P6.UP / "deephf_3enzyme_derived.csv", index_col=0)
    hf = raw.loc[pd.Index(D["seq"]), "hf"].values.astype(np.float32)
    wt = D["y1"]
    te = np.where(D["split"] == 2)[0]
    y2_te, wt_te, hf_te = D["y2"][te], wt[te], np.asarray(hf)[te]

    R = {"seed_bases": SEED_BASES, "runs": []}
    t0 = time.time()
    print(f"[env] device = {P6.DEV} | seed bases = {SEED_BASES}", flush=True)

    for b in SEED_BASES:
        rec = {"seed_base": b}
        for tag, frac in [("full", 1.0), ("sub10", 0.10)]:
            idx_all = np.where(D["split"] == 0)[0]
            if frac < 1.0:
                rs = np.random.default_rng(20260830)      # 원본과 동일 — 고정
                idx_tr = np.sort(rs.choice(idx_all, int(len(idx_all) * frac), replace=False))
            else:
                idx_tr = idx_all

            p_wt, _ = P6.train_single(wt, D, idx_tr, seed=b + 1)
            p_hf, _ = P6.train_single(hf, D, idx_tr, seed=b + 2)
            p_st, _ = P6.train_single(D["y2"], D, idx_tr, seed=b + 3)
            p_mt1, p_mt2, _ = P6.train_multi(D, idx_tr, seed=b + 4)

            preds = {"BASE": np.clip(p_hf / np.maximum(p_wt, 1e-6), 0, P6.CLIP),
                     "ST": p_st, "MT": p_mt2}
            overall = {k: float(spearmanr(v, y2_te)[0]) for k, v in preds.items()}

            q = np.quantile(wt_te, [0.25, 0.5, 0.75])
            strata = np.digitize(wt_te, q)
            strat = [{k: float(spearmanr(v[strata == s], y2_te[strata == s])[0])
                      for k, v in preds.items()} for s in range(4)]

            rec[tag] = dict(overall=overall, strata=strat,
                            aux=dict(base_wt=float(spearmanr(p_wt, wt_te)[0]),
                                     base_hf=float(spearmanr(p_hf, hf_te)[0]),
                                     mt_y1=float(spearmanr(p_mt1, wt_te)[0])))
            print(f"  b={b:2d} {tag:5s} | BASE {overall['BASE']:.4f}  ST {overall['ST']:.4f}  "
                  f"MT {overall['MT']:.4f} | MT-BASE {overall['MT']-overall['BASE']:+.4f}  "
                  f"MT-ST {overall['MT']-overall['ST']:+.4f} | {time.time()-t0:5.0f}s", flush=True)
        R["runs"].append(rec)

    json.dump(R, open(P6.ROOT / "out/phase6_seed_sweep.json", "w"), indent=2)
    print("\n[save] out/phase6_seed_sweep.json", flush=True)

    # ---- 즉시 판정 (사전등록 D2 ~ D5)
    n = len(SEED_BASES)
    g = lambda r, tag, m: r[tag]["overall"][m]
    d_MB = np.array([g(r, "full", "MT") - g(r, "full", "BASE") for r in R["runs"]])
    d_MS = np.array([g(r, "full", "MT") - g(r, "full", "ST") for r in R["runs"]])
    d_SB = np.array([g(r, "full", "ST") - g(r, "full", "BASE") for r in R["runs"]])

    def verdict(d, name):
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        ok = abs(m) > 2 * se
        print(f"  {name:10s} 평균 {m:+.4f} ± SE {se:.4f}  → "
              f"{'구분됨 (|d| > 2SE)' if ok else '구분 불가 (|d| <= 2SE)'}")
        return dict(mean=float(m), se=float(se), distinguishable=bool(ok),
                    values=[float(x) for x in d])

    print("\n===== D2/D3 쌍대 비교 (full 조건, n = %d) =====" % n)
    V = dict(MT_minus_BASE=verdict(d_MB, "MT - BASE"),
             MT_minus_ST=verdict(d_MS, "MT - ST"),
             ST_minus_BASE=verdict(d_SB, "ST - BASE"))

    q1max = sum(int(np.argmax([s["MT"] - s["BASE"] for s in r["full"]["strata"]]) == 0)
                for r in R["runs"])
    sub_gt = sum(int((g(r, "sub10", "MT") - g(r, "sub10", "BASE")) >
                     (g(r, "full", "MT") - g(r, "full", "BASE"))) for r in R["runs"])
    rule = lambda c: "지지" if c >= 4 else ("기각" if c <= 2 else "판정 불가")
    print(f"\n===== D4/D5 사전 예측 재판정 (5회 중) =====")
    print(f"  P2  Q1 최대 {q1max}/5  → {rule(q1max)}")
    print(f"  P3  10%>전체 {sub_gt}/5  → {rule(sub_gt)}")
    V["P2_q1_max_count"] = q1max
    V["P3_sub10_larger_count"] = sub_gt
    V["P2_verdict"], V["P3_verdict"] = rule(q1max), rule(sub_gt)
    R["seed_verdict"] = V
    json.dump(R, open(P6.ROOT / "out/phase6_seed_sweep.json", "w"), indent=2)
