# claude/data/B1_qscan_fullsample.py 재실행 (CSV 경로만 이 환경에 맞춤)
import numpy as np, pandas as pd, math, json
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _env
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = pd.read_csv(_env.data_dir() / 'deephf_3enzyme_derived.csv', index_col=0)
print(f"행수 {len(D):,}")
Y1 = D['wt'].to_numpy(float)
Y2 = np.clip(D['ratio_hf'].to_numpy(float), None, 1.5)
# 기록값 0.4842 는 1.5 클리핑 '후' 평균이다 (클리핑 전은 0.4847).
# 게이트가 Y2 만 클리핑하고 Y2v 를 클리핑하지 않아 2026-10-31 에 불일치로 잡혔다.
Y2v = np.clip(D['ratio_esp'].to_numpy(float), None, 1.5)
print("\n=== 게이트: 기록값과 대조 ===")
for nm, got, exp in [("N", len(D), 53937), ("Y1 평균", Y1.mean(), 0.7248),
                     ("Y2 평균", Y2.mean(), 0.6666), ("Y2v 평균", Y2v.mean(), 0.4842),
                     ("클리핑 건수", int((D['ratio_hf']>1.5).sum()), 161),
                     ("비율>1 건수", int((D['ratio_hf']>1).sum()), 2049)]:
    ok = abs(got-exp) < (0.0001 if isinstance(exp,float) else 1)
    print(f"  {nm:12} {got:>12.4f}  기록 {exp:>10}  {'OK' if ok else '<<< 불일치'}")
r_p = stats.pearsonr(Y1,Y2)[0]; r_s = stats.spearmanr(Y1,Y2)[0]
print(f"  {'Pearson':12} {r_p:>12.4f}  기록 {-0.1156:>10}  {'OK' if abs(r_p+0.1156)<0.0002 else '<<< 불일치'}")
print(f"  {'Spearman':12} {r_s:>12.4f}  기록 {+0.0253:>10}  {'OK' if abs(r_s-0.0253)<0.0002 else '<<< 불일치'}")

N = len(Y1)
QS = [0.30,0.25,0.20,0.18,0.15,0.12,0.10,0.09,0.08,0.07,0.05,0.04,0.03]
rk1 = stats.rankdata(Y1); rk2 = stats.rankdata(Y2)
def joint(r1, r2, q, n):
    k = int(round(n*q)); return int(np.sum((r1 > n-k) & (r2 > n-k)))
obs = {q: joint(rk1, rk2, q, N) for q in QS}
R = 2*math.sin(math.pi*r_s/6)
print(f"\nSpearman {r_s:.4f} -> copula Pearson r = {R:.4f}")
MC, SEED = 2000, 20260906
rng = np.random.default_rng(SEED); L = np.linalg.cholesky([[1,R],[R,1]])
acc = {q: np.empty(MC) for q in QS}
for i in range(MC):
    z = rng.standard_normal((N,2)) @ L.T
    a1 = stats.rankdata(z[:,0]); a2 = stats.rankdata(z[:,1])
    for q in QS: acc[q][i] = joint(a1, a2, q, N)
REF = {0.30:17.47,0.25:16.67,0.20:14.41,0.18:12.09,0.15:8.48,0.12:2.28,0.10:-2.40,
       0.09:-4.47,0.08:-6.10,0.07:-8.04,0.05:-10.65,0.04:-9.71,0.03:-7.59}
REF_OBS = {0.30:5890,0.25:4261,0.20:2818,0.18:2270,0.15:1543,0.12:891,0.10:532,
           0.09:388,0.08:269,0.07:164,0.05:27,0.04:5,0.03:0}
REF_NULL = {0.30:(5026.3,49.4),0.25:(3515.1,44.7),0.20:(2269.9,38.0),0.18:(1846.5,35.0),
            0.15:(1292.4,29.5),0.12:(834.6,24.7),0.10:(583.9,21.6),0.09:(475.0,19.5),
            0.08:(377.7,17.8),0.07:(290.8,15.8),0.05:(150.3,11.6),0.04:(97.2,9.5),0.03:(55.4,7.3)}
print(f"\n{'q':>6} {'관측':>7} {'기록관측':>8} {'귀무평균':>9} {'기록':>8} {'sd':>6} {'기록':>6} {'z':>8} {'지시서z':>8} {'차':>7}")
rows=[]; bad=0
for q in QS:
    m, s = acc[q].mean(), acc[q].std(ddof=1); z = (obs[q]-m)/s
    dm, ds = REF_NULL[q]
    d = z - REF[q]
    flag = '' if abs(d) < 0.02 else '  <<<'
    if abs(d) >= 0.02: bad += 1
    print(f"{q:6.2f} {obs[q]:7d} {REF_OBS[q]:8d} {m:9.1f} {dm:8.1f} {s:6.1f} {ds:6.1f} {z:+8.2f} {REF[q]:+8.2f} {d:+7.3f}{flag}")
    rows.append(dict(q=q, observed=obs[q], null_mean=float(m), null_sd=float(s), z=float(z)))
print(f"\n관측 수 전부 일치: {all(obs[q]==REF_OBS[q] for q in QS)}")
print(f"z 가 지시서와 0.02 이상 다른 지점: {bad}개")
json.dump(dict(n=N, seed=SEED, reps=MC, spearman=float(r_s), copula_r=float(R), rows=rows),
          open(ROOT / 'out/B1_fullsample.json','w'), ensure_ascii=False, indent=1)
print("saved out/B1_fullsample.json")
