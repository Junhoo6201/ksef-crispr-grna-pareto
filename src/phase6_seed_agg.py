"""
Phase 6 시드 스윕 집계 — 그림 6 의 입력 (2026-09-13)
====================================================
out/phase6_seed_sweep.json (시드 베이스 5개 × {full, sub10}) 을 묶어
평균과 **표본표준편차(ddof = 1)** 를 낸다.

출력: out/phase6_seed_agg.json  (새 파일)
※ out/phase6_baseline.json 은 원 실행의 근거이므로 건드리지 않는다.

`python3 src/phase6_seed_agg.py` 로 실행하면 회신 §3 검산값과 대조한다.
"""
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "out/phase6_seed_sweep.json"
DST = ROOT / "out/phase6_seed_agg.json"
METHODS = ("BASE", "ST", "MT")

# 도판-회신-2026-09-13 §3 검산값 (평균, sd)
CHECK = {
    "full":  {"BASE": (0.6502, 0.0242), "ST": (0.7039, 0.0023), "MT": (0.7019, 0.0096)},
    "sub10": {"BASE": (0.5107, 0.0161), "ST": (0.5470, 0.0076), "MT": (0.5627, 0.0057)},
}


def aggregate(path=SRC):
    J = json.load(open(path, encoding="utf-8"))
    runs, bases = J["runs"], J["seed_bases"]
    out = {"seed_bases": bases, "n": len(runs), "ddof": 1}
    for cond in ("full", "sub10"):
        ov = {m: np.array([r[cond]["overall"][m] for r in runs], float) for m in METHODS}
        gap = np.array([[s["MT"] - s["BASE"] for s in r[cond]["strata"]] for r in runs], float)
        out[cond] = {
            "overall": {m: {"mean": float(ov[m].mean()), "sd": float(ov[m].std(ddof=1)),
                            "values": [float(v) for v in ov[m]]} for m in METHODS},
            "quartile_gap_MT_minus_BASE": {
                "mean": [float(v) for v in gap.mean(axis=0)],
                "sd": [float(v) for v in gap.std(axis=0, ddof=1)],
                "per_seed": gap.tolist(),
                "argmax_quartile": [int(np.argmax(row)) + 1 for row in gap],
            },
        }
    return out


def check(agg):
    bad = []
    for cond, d in CHECK.items():
        for m, (em, es) in d.items():
            g = agg[cond]["overall"][m]
            if round(g["mean"], 4) != round(em, 4):
                bad.append((cond, m, "평균", round(g["mean"], 4), em))
            if round(g["sd"], 4) != round(es, 4):
                bad.append((cond, m, "sd", round(g["sd"], 4), es))
    return bad


if __name__ == "__main__":
    agg = aggregate()
    bad = check(agg)
    for cond in ("full", "sub10"):
        print(f"[{cond}]  " + "  ".join(
            f"{m} {agg[cond]['overall'][m]['mean']:.4f} ± {agg[cond]['overall'][m]['sd']:.4f}"
            for m in METHODS))
        q = agg[cond]["quartile_gap_MT_minus_BASE"]
        print("        사분위 격차 " + "  ".join(
            f"Q{i+1} {q['mean'][i]:+.4f} ± {q['sd'][i]:.4f}" for i in range(4)))
        print(f"        시드별 최대 사분위 {q['argmax_quartile']}")
    if bad:
        print("\n[불일치]")
        for row in bad:
            print("  ", row)
        sys.exit(1)
    print("\n[검산] full·sub10 × BASE·ST·MT 12개 값 전부 일치 (ddof=1)")
    json.dump(agg, open(DST, "w"), indent=2, ensure_ascii=False)
    print(f"[save] out/{DST.name}")
