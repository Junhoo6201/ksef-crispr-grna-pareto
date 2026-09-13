"""
Phase 2 시드 스윕 — 사전등록 부록 (2026-09-12)
=============================================
커밋된 phase2_model.py 를 수정하지 않고 run() 만 import 하여 호출한다.
λ 11개 × 시드 5개 = 55개 모델을 학습하고 성능만 기록한다.

출력: out/phase2_seed_sweep.json  (새 파일)
※ out/phase2_lambda_sweep.json 과 out/model_lam050.pt 는 건드리지 않는다.
   원고 정본 수치와 Phase 3~7 의 입력이 그 두 파일이기 때문이다.
"""
import json, pathlib, sys, time
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import phase2_model as P2

SEEDS = [0, 1, 2, 3, 4]                                   # 실행 전 고정
LAMS = [round(float(x), 1) for x in np.arange(0, 1.01, 0.1)]

if __name__ == "__main__":
    D = dict(np.load(P2.ROOT / "out/dataset.npz", allow_pickle=True))
    rows, t0 = [], time.time()
    print(f"[env] device = {P2.DEV} | seeds = {SEEDS} | lambdas = {len(LAMS)}", flush=True)
    for s in SEEDS:
        for lam in LAMS:
            _, r = P2.run(lam, D, seed=s)
            r["seed"] = s
            rows.append(r)
            print(f"seed {s} | lam {lam:.1f} | eff {r['sp_eff']:+.4f} "
                  f"fid {r['sp_fid']:+.4f} | ep {r['epochs_run']:2d} | {time.time()-t0:5.0f}s",
                  flush=True)
    json.dump(rows, open(P2.ROOT / "out/phase2_seed_sweep.json", "w"), indent=2)
    print("[save] out/phase2_seed_sweep.json", flush=True)

    # ---- 즉시 요약 (사전등록 D1)
    def col(lam, key):
        return np.array([r[key] for r in rows if abs(r["lam"] - lam) < 1e-9])
    print("\n===== λ별 시드 평균 ± 표준편차 (n = %d) =====" % len(SEEDS))
    print(f"{'lam':>5s} {'Y1 mean':>9s} {'Y1 sd':>8s} {'Y2 mean':>9s} {'Y2 sd':>8s}")
    for lam in LAMS:
        e, f = col(lam, "sp_eff"), col(lam, "sp_fid")
        print(f"{lam:5.1f} {e.mean():+9.4f} {e.std(ddof=1):8.4f} "
              f"{f.mean():+9.4f} {f.std(ddof=1):8.4f}")
    e05, f05 = col(0.5, "sp_eff"), col(0.5, "sp_fid")
    print(f"\n[D1] λ=0.5 잡음 기준  s1 = {e05.std(ddof=1):.4f}  s2 = {f05.std(ddof=1):.4f}"
          f"   (기존 2회 관측차: 0.0025 / 0.0112)")
