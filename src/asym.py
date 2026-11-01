"""#6 — Y2v 에서 전선은 확대되는데 꼬리 고갈은 없는 비대칭의 기전 탐색
사후 탐색(post-hoc)이다. 원고의 사전 설계 격자(0.30~0.03)를 바꾸지 않는다."""
import numpy as np, pandas as pd, math, json, pathlib
from scipy import stats
U = pathlib.Path(__file__).resolve().parents[1]   # 저장소 뿌리
D = pd.read_csv(U/'data/deephf_3enzyme_derived.csv', index_col=0)
Y1 = D['wt'].to_numpy(float); N = len(Y1)
T = {'Y2': np.clip(D['ratio_hf'].to_numpy(float), None, 1.5),
     'Y2v': np.clip(D['ratio_esp'].to_numpy(float), None, 1.5)}

def front_exact(u, w):
    o=np.argsort(-u,kind='mergesort'); us,ws=u[o],w[o]; n=len(us)
    st=np.r_[0, np.flatnonzero(np.diff(us)!=0)+1]
    gid=np.zeros(n,int); gid[st[1:]]=1; gid=np.cumsum(gid)
    m_eq=np.maximum.reduceat(ws,st); m_gt=np.r_[-np.inf,np.maximum.accumulate(m_eq)[:-1]]
    return int((~((m_gt[gid]>=ws)|(m_eq[gid]>ws))).sum())

print('=== T4. 귀무 전선은 상관에 얼마나 민감한가 (관측값과 분리해서 본다) ===')
print(f'{"ρs":>8}{"copula r":>10}{"귀무 전선 평균":>14}{"sd":>7}')
rows_t4=[]
for rho in (0.00, 0.0253, 0.10, 0.2625, 0.40):
    r = 2*math.sin(math.pi*rho/6); L=np.linalg.cholesky([[1,r],[r,1]])
    rng=np.random.default_rng(20260906)
    F=np.array([front_exact(stats.rankdata(z[:,0])/N, stats.rankdata(z[:,1])/N)
                for z in (rng.standard_normal((N,2))@L.T for _ in range(400))])
    rows_t4.append(dict(rho=rho, r=r, mean=float(F.mean()), sd=float(F.std(ddof=1))))
    print(f'{rho:+8.4f}{r:+10.4f}{F.mean():14.2f}{F.std(ddof=1):7.2f}')
obs = {k: front_exact(stats.rankdata(Y1)/N, stats.rankdata(v)/N) for k,v in T.items()}
print(f'\n관측 전선: Y2 {obs["Y2"]}  ·  Y2v {obs["Y2v"]}   → 두 관측값은 거의 같다')
print('즉 Y2v 의 z 가 큰 것은 관측이 커서가 아니라 귀무가 작아서다.')

print('\n=== T5. 사전 격자 밖(q < 0.03)까지 분위 스캔을 연장 ===')
QS=[0.03,0.02,0.015,0.01,0.005,0.002]
REPS=1000
print(f'{"q":>7}{"k":>7} | ' + ' | '.join(f'{n:>26}' for n in T))
res={}
for nm,Y2 in T.items():
    rs=stats.spearmanr(Y1,Y2)[0]; r=2*math.sin(math.pi*rs/6)
    L=np.linalg.cholesky([[1,r],[r,1]]); rng=np.random.default_rng(20260906)
    r1=stats.rankdata(Y1); r2=stats.rankdata(Y2)
    obs_q={q:int(((r1>N-round(N*q))&(r2>N-round(N*q))).sum()) for q in QS}
    null={q:np.empty(REPS) for q in QS}
    for b in range(REPS):
        z=rng.standard_normal((N,2))@L.T
        q1=stats.rankdata(z[:,0]); q2=stats.rankdata(z[:,1])
        for q in QS:
            k=round(N*q); null[q][b]=((q1>N-k)&(q2>N-k)).sum()
    res[nm]={}
    for q in QS:
        m,s=null[q].mean(), null[q].std(ddof=1)
        res[nm][str(q)]=dict(k=round(N*q), obs=obs_q[q], mean=float(m), sd=float(s),
                             z=float((obs_q[q]-m)/s) if s>0 else None)
for q in QS:
    line=f'{q:7.3f}{round(N*q):7,} | '
    for nm in T:
        d=res[nm][str(q)]
        zz = f'{d["z"]:+6.2f}' if d['z'] is not None else '   n/a'
        line += f'관측 {d["obs"]:4d} 귀무 {d["mean"]:7.2f} z {zz} | '
    print(line)
json.dump(dict(t4=rows_t4, obs_front=obs, qscan_ext=res, reps=REPS, seed=20260906),
          open(U/'out/asym.json','w'), ensure_ascii=False, indent=1)
print(f'\n[save] {U}/out/asym.json')
