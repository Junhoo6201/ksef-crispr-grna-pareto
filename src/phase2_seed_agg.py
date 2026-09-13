"""
Phase 2 시드 스윕 집계 — 그림 2·10 의 입력 (2026-09-13)
=======================================================
out/phase2_seed_sweep.json (λ 11 × 시드 5 = 55개) 을 λ 별로 묶어
평균과 **표본표준편차(ddof = 1)** 를 낸다. ddof=0 이 아니다.

출력: out/phase2_seed_agg.json  (새 파일)
※ out/phase2_lambda_sweep.json 은 표 1(원 실행)의 근거이므로 건드리지 않는다.

`python3 src/phase2_seed_agg.py --check` 로 실행하면 지시서 §4 검산표와 대조한다.
"""
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "out/phase2_seed_sweep.json"
DST = ROOT / "out/phase2_seed_agg.json"

# 지시서 2026-09-13 §4 검산표 (λ, Y1 평균, Y1 sd, Y2 평균, Y2 sd)
CHECK = [
    (0.0, -0.1454, 0.3698, 0.6996, 0.0130), (0.1, 0.8267, 0.0026, 0.7047, 0.0039),
    (0.2, 0.8350, 0.0022, 0.7135, 0.0023), (0.3, 0.8445, 0.0033, 0.7113, 0.0072),
    (0.4, 0.8455, 0.0065, 0.7117, 0.0028), (0.5, 0.8518, 0.0009, 0.7069, 0.0038),
    (0.6, 0.8499, 0.0086, 0.7050, 0.0104), (0.7, 0.8514, 0.0068, 0.7067, 0.0103),
    (0.8, 0.8553, 0.0024, 0.7060, 0.0025), (0.9, 0.8506, 0.0082, 0.6757, 0.0323),
    (1.0, 0.8481, 0.0103, -0.0454, 0.0414),
]


def aggregate(path=SRC):
    rows = json.load(open(path, encoding="utf-8"))
    lams = sorted({round(float(r["lam"]), 10) for r in rows})
    out = []
    for lam in lams:
        g = [r for r in rows if abs(float(r["lam"]) - lam) < 1e-9]
        e = np.array([r["sp_eff"] for r in g], float)
        f = np.array([r["sp_fid"] for r in g], float)
        out.append(dict(lam=lam, n=len(g),
                        eff_mean=float(e.mean()), eff_sd=float(e.std(ddof=1)),
                        fid_mean=float(f.mean()), fid_sd=float(f.std(ddof=1)),
                        seeds=sorted(int(r["seed"]) for r in g)))
    return out


def check(agg):
    bad = []
    for a, (lam, em, es, fm, fs) in zip(agg, CHECK):
        assert abs(a["lam"] - lam) < 1e-9, (a["lam"], lam)
        for name, got, exp in (("Y1 평균", a["eff_mean"], em), ("Y1 sd", a["eff_sd"], es),
                               ("Y2 평균", a["fid_mean"], fm), ("Y2 sd", a["fid_sd"], fs)):
            if round(got, 4) != round(exp, 4):
                bad.append((lam, name, round(got, 4), exp))
    return bad


if __name__ == "__main__":
    agg = aggregate()
    bad = check(agg)
    print(f"{'lam':>5s} {'Y1 mean':>9s} {'Y1 sd':>8s} {'Y2 mean':>9s} {'Y2 sd':>8s}  n")
    for a in agg:
        print(f"{a['lam']:5.1f} {a['eff_mean']:+9.4f} {a['eff_sd']:8.4f} "
              f"{a['fid_mean']:+9.4f} {a['fid_sd']:8.4f}  {a['n']}")
    if bad:
        print("\n[불일치]")
        for lam, name, got, exp in bad:
            print(f"  λ={lam:.1f} {name}: 산출 {got} vs 검산표 {exp}")
        sys.exit(1)
    print("\n[검산] 11개 λ 전부 일치 (ddof=1)")
    json.dump({"source": SRC.name, "ddof": 1, "rows": agg},
              open(DST, "w"), indent=2, ensure_ascii=False)
    print(f"[save] out/{DST.name}")
