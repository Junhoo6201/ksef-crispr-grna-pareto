"""#6 교차확인 — (1) 다른 시드, (2) 이변량 정규 사분면 확률 해석해"""
import numpy as np, pandas as pd, math, pathlib
from scipy import stats
U = pathlib.Path(__file__).resolve().parents[1]   # 저장소 뿌리
D = pd.read_csv(U/'data/deephf_3enzyme_derived.csv', index_col=0)
Y1=D['wt'].to_numpy(float); N=len(Y1)
T={'Y2':np.clip(D['ratio_hf'].to_numpy(float),None,1.5),
   'Y2v':np.clip(D['ratio_esp'].to_numpy(float),None,1.5)}
QS=[0.03,0.02,0.015,0.01,0.005]
for nm,Y2 in T.items():
    rs=stats.spearmanr(Y1,Y2)[0]; r=2*math.sin(math.pi*rs/6)
    mvn=stats.multivariate_normal(mean=[0,0],cov=[[1,r],[r,1]])
    r1=stats.rankdata(Y1); r2=stats.rankdata(Y2)
    L=np.linalg.cholesky([[1,r],[r,1]]); rng=np.random.default_rng(20260907)
    null={q:np.empty(400) for q in QS}
    for b in range(400):
        z=rng.standard_normal((N,2))@L.T
        q1=stats.rankdata(z[:,0]); q2=stats.rankdata(z[:,1])
        for q in QS:
            k=round(N*q); null[q][b]=((q1>N-k)&(q2>N-k)).sum()
    print(f'\n=== {nm} (ρs {rs:+.4f}) ===')
    print(f'{"q":>7}{"관측":>6}{"해석해 기대":>12}{"MC평균(0907)":>13}{"sd":>7}{"z(0907)":>9}')
    for q in QS:
        k=round(N*q); obs=int(((r1>N-k)&(r2>N-k)).sum())
        zq=stats.norm.ppf(1-q)
        p=1-2*stats.norm.cdf(zq)+mvn.cdf([zq,zq])
        m,s=null[q].mean(),null[q].std(ddof=1)
        print(f'{q:7.3f}{obs:6d}{N*p:12.2f}{m:13.2f}{s:7.2f}{(obs-m)/s:+9.2f}')
