# Y2v(eSpCas9) 분위 스캔 — 4.4절과 같은 절차, 모델 불요
# 원고 격자: 0.30·0.25·0.20·0.18·0.15·0.12·0.10·0.09·0.08·0.07·0.05·0.04·0.03
import numpy as np, pandas as pd, math, scipy, scipy.stats as st, json, pathlib
SEED, REPS = 20260906, 2000
QS = [0.30,0.25,0.20,0.18,0.15,0.12,0.10,0.09,0.08,0.07,0.05,0.04,0.03]
D = pd.read_csv(pathlib.Path(__file__).resolve().parents[1]/'data'/'deephf_3enzyme_derived.csv', index_col=0)
Y1 = D['wt'].to_numpy(float)
Y2v = np.clip(D['ratio_esp'].to_numpy(float), None, 1.5)
N = len(Y1)
print(f'numpy {np.__version__} scipy {scipy.__version__}  N={N:,}')
print(f'Y1 평균 {Y1.mean():.4f}   Y2v 평균(절단후) {Y2v.mean():.4f}')
rs = st.spearmanr(Y1, Y2v).statistic
r  = 2*math.sin(math.pi*rs/6)
print(f'Spearman rho_s = {rs:+.4f}  → copula r = {r:+.4f}')
r1 = st.rankdata(Y1); r2 = st.rankdata(Y2v)
obs = {}
for q in QS:
    k = round(N*q)
    obs[q] = int(np.intersect1d(np.argsort(-r1)[:k], np.argsort(-r2)[:k]).size)
rng = np.random.default_rng(SEED)
L = np.linalg.cholesky(np.array([[1.0, r],[r, 1.0]]))
cnt = {q: np.empty(REPS) for q in QS}
for b in range(REPS):
    z = (L @ rng.standard_normal((2, N)))
    a1 = np.argsort(-z[0]); a2 = np.argsort(-z[1])
    for q in QS:
        k = round(N*q)
        cnt[q][b] = np.intersect1d(a1[:k], a2[:k]).size
print(f'\n{"q":>6} {"k":>7} {"관측":>7} {"귀무평균":>10} {"sd":>7} {"z":>8}')
out={}
for q in QS:
    m, s = cnt[q].mean(), cnt[q].std(ddof=1)
    z = (obs[q]-m)/s
    out[str(q)] = dict(k=round(N*q), observed=obs[q], null_mean=float(m), null_sd=float(s), z=float(z))
    print(f'{q:6.2f} {round(N*q):7,} {obs[q]:7,} {m:10.1f} {s:7.2f} {z:+8.2f}')
json.dump({'n':N,'seed':SEED,'reps':REPS,'spearman':float(rs),'copula_r':float(r),'rows':out},
          open(pathlib.Path(__file__).resolve().parents[1]/'out'/'esp_qscan_null.json','w'), indent=1)
zs=[out[str(q)]['z'] for q in QS]
cross=[(QS[i],QS[i+1]) for i in range(len(QS)-1) if zs[i]>0>=zs[i+1]]
print(f'\nz 가 0 을 지나는 구간: {cross}')
