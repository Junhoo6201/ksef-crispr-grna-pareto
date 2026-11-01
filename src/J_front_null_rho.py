"""4.7절 보조 — 귀무 전선 크기가 ρₛ에 따라 줄어드는 정도.

원고 4.7절은 Y2 와 Y2v 의 귀무 전선 차이(11.07 대 8.25)가 효소의 세기 차이가
아니라 전역 상관 차이에서 온다고 서술한다. 그 근거가 되는 ρₛ–귀무전선 관계를
여기서 산출한다. 관측 자료를 쓰지 않는 순수 시뮬레이션이다.

교차확인(지침 §3-2): ρₛ = 0 일 때 귀무 전선의 기대값은 조화수
Hₙ = Σ_{i=1..n} 1/i 와 일치해야 한다(독립 좌표 n점의 파레토 전선 크기 기대값의
정확해). 시뮬레이션과 해석해를 함께 출력하여 대조한다.
"""
import numpy as np, math, json, pathlib, sys
from scipy import stats

U = pathlib.Path(__file__).resolve().parents[1]
OUT = sys.argv[1] if len(sys.argv) > 1 else U / 'out/front_null_rho.json'

N    = 53937
REPS = 2000
SEED = 20260906
RHOS = [0.0, 0.0253, 0.2625, 0.40]

def front_fast(u, w):                 # 연속 귀무표본 — 동점 없음
    o = np.argsort(-u, kind='mergesort'); ws = w[o]
    run = np.maximum.accumulate(ws)
    return int(np.concatenate(([True], ws[1:] > run[:-1])).sum())

Hn = sum(1.0/i for i in range(1, N+1))
print(f"n = {N:,}   reps = {REPS}   seed = {SEED}")
print(f"해석해  Hₙ = Σ1/i = {Hn:.4f}   (독립 좌표일 때 전선 크기 기대값)\n")
print(f"{'ρₛ':>8}{'코퓰러 r':>11}{'귀무 평균':>12}{'sd':>8}{'reps':>7}")

rows = {}
for rs in RHOS:
    r = 2*math.sin(math.pi*rs/6)
    L = np.linalg.cholesky([[1, r], [r, 1]])
    rng = np.random.default_rng(SEED)
    F = np.empty(REPS, int)
    for i in range(REPS):
        z = rng.standard_normal((N, 2)) @ L.T
        F[i] = front_fast(stats.rankdata(z[:, 0])/N, stats.rankdata(z[:, 1])/N)
    rows[f"{rs}"] = dict(rho_s=rs, copula_r=float(r), null_mean=float(F.mean()),
                         null_sd=float(F.std(ddof=1)), reps=REPS)
    print(f"{rs:>8.4f}{r:>11.4f}{F.mean():>12.2f}{F.std(ddof=1):>8.2f}{REPS:>7}")

m0 = rows["0.0"]["null_mean"]
print(f"\n교차확인  ρₛ=0 시뮬레이션 {m0:.4f}  vs  해석해 Hₙ {Hn:.4f}"
      f"   차이 {m0-Hn:+.4f} ({abs(m0-Hn)/Hn*100:.2f}%)")
mc_se = rows["0.0"]["null_sd"]/math.sqrt(REPS)
print(f"          몬테카를로 표준오차 = {mc_se:.4f}"
      f"  → 차이는 {abs(m0-Hn)/mc_se:.2f} SE  "
      f"{'(일치)' if abs(m0-Hn) < 3*mc_se else '(불일치 — 보고 필요)'}")

json.dump(dict(n=N, reps=REPS, seed=SEED, Hn=Hn, rows=rows,
               crosscheck=dict(sim_rho0=m0, analytic_Hn=Hn, diff=m0-Hn,
                               mc_se=mc_se, se_units=abs(m0-Hn)/mc_se)),
          open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"\n저장: {OUT}")
