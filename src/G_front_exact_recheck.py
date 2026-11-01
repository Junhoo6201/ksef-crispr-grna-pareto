"""4.7절 전선 크기 전면 재집계 — 약지배 정확 집계로 표 3 전 행 + Y2v
귀무표본은 연속분포로 동점이 없으므로(검증: 동점 없을 때 네 구현 50/50 일치)
빠른 구현을 쓰고, 관측값만 정확 집계를 쓴다."""
import numpy as np, torch, torch.nn as nn, math, json, pathlib
from scipy import stats
U = pathlib.Path(__file__).resolve().parents[1]

def front_exact(u, w):                      # 동점 허용 · 벡터화
    o = np.argsort(-u, kind='mergesort'); us, ws = u[o], w[o]; n = len(us)
    starts = np.r_[0, np.flatnonzero(np.diff(us) != 0) + 1]
    gid = np.zeros(n, int); gid[starts[1:]] = 1; gid = np.cumsum(gid)
    m_eq = np.maximum.reduceat(ws, starts)              # 그룹별 w 최대
    m_gt = np.r_[-np.inf, np.maximum.accumulate(m_eq)[:-1]]   # u 가 더 큰 그룹들의 최대
    dom = (m_gt[gid] >= ws) | (m_eq[gid] > ws)
    return int((~dom).sum())

def front_fast(u, w):                       # 동점 없을 때만 사용
    o = np.argsort(-u, kind='mergesort'); ws = w[o]
    run = np.maximum.accumulate(ws)
    return int(np.concatenate(([True], ws[1:] > run[:-1])).sum())

class MultiTaskCNN(nn.Module):
    def __init__(self, n_bio=11, ch=128, emb=64):
        super().__init__()
        self.trunk=nn.Sequential(nn.Conv1d(4,ch,3,padding=1),nn.BatchNorm1d(ch),nn.ReLU(),
            nn.Conv1d(ch,ch,5,padding=2),nn.BatchNorm1d(ch),nn.ReLU(),
            nn.Conv1d(ch,ch,7,padding=3),nn.BatchNorm1d(ch),nn.ReLU())
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.merge=nn.Sequential(nn.Linear(ch+n_bio,emb*2),nn.ReLU(),nn.Dropout(0.2),
                                 nn.Linear(emb*2,emb),nn.ReLU())
        h=lambda: nn.Sequential(nn.Linear(emb,32),nn.ReLU(),nn.Linear(32,1))
        self.head_eff,self.head_fid=h(),h()
    def forward(self,xs,xb):
        z=self.merge(torch.cat([self.pool(self.trunk(xs)).squeeze(-1),xb],1))
        return self.head_eff(z).squeeze(-1), self.head_fid(z).squeeze(-1)

# 정확 집계 구현 검증 (동점 많은 소규모 vs 무차별 대입)
def brute(u,w):
    n=len(u); d=np.zeros(n,bool)
    for i in range(n):
        d[i]=bool((((u>=u[i])&(w>=w[i]))&((u>u[i])|(w>w[i]))).any())
    return int((~d).sum())
rng=np.random.default_rng(1); bad=0
for _ in range(300):
    a=rng.integers(0,8,60).astype(float); b=rng.integers(0,8,60).astype(float)
    bad += front_exact(a,b)!=brute(a,b)
print(f'[검증] front_exact vs 무차별대입 300회 불일치 {bad}건')

torch.set_num_threads(4)
D = dict(np.load(U/'out/dataset.npz', allow_pickle=True))
y1,y2,y2v,split = (D[k].astype(float) for k in ('y1','y2','y2v')), None, None, None
y1,y2,y2v = D['y1'].astype(float), D['y2'].astype(float), D['y2v'].astype(float)
split = D['split']; te = split==2
m = MultiTaskCNN(D['X_bio'].shape[1]); m.load_state_dict(torch.load(U/'out/model_lam050.pt', map_location='cpu')); m.eval()
with torch.no_grad():
    p1,p2 = m(torch.tensor(D['X_seq'][te]), torch.tensor(D['X_bio'][te]))
p1,p2 = p1.numpy().astype(float), p2.numpy().astype(float)

def run(nm,u,w,reps=2000,seed=20260906):
    n=len(u); a1,a2=stats.rankdata(u)/n, stats.rankdata(w)/n
    fe=front_exact(a1,a2); fh=front_fast(a1,a2)
    rs=stats.spearmanr(u,w)[0]; r=2*math.sin(math.pi*rs/6)
    L=np.linalg.cholesky([[1,r],[r,1]]); rng=np.random.default_rng(seed)
    F=np.empty(reps,int)
    for i in range(reps):
        z=rng.standard_normal((n,2))@L.T
        F[i]=front_fast(stats.rankdata(z[:,0])/n, stats.rankdata(z[:,1])/n)
    z_=(fe-F.mean())/F.std(ddof=1); p=(F>=fe).mean()
    rng=np.random.default_rng(20260906); FS=[]
    for _ in range(30):
        j1=stats.rankdata(u+rng.normal(0,1e-12,n))/n; j2=stats.rankdata(w+rng.normal(0,1e-12,n))/n
        FS.append(front_exact(j1,j2))
    FS=np.array(FS)
    out=dict(n=n, exact=fe, manuscript_code=fh, null_mean=float(F.mean()), null_sd=float(F.std(ddof=1)),
             z=float(z_), p=float(p), ge=int((F>=fe).sum()), reps=reps, spearman=float(rs),
             Hn=sum(1.0/i for i in range(1,n+1)), tie_range=[int(FS.min()),int(FS.max())],
             p_worst=float((F>=FS.min()).mean()), z_code=float((fh-F.mean())/F.std(ddof=1)))
    print(f'{nm:<22}{fe:6d}{fh:8d}{F.mean():9.2f}{F.std(ddof=1):6.2f}{z_:+7.2f}{p:8.4f}'
          f'   동점분리 {FS.min()}–{FS.max()} 최악p={out["p_worst"]:.4f}  Hn={out["Hn"]:.2f}')
    return out

print(f'\n{"조건":<22}{"exact":>6}{"원고":>8}{"귀무":>9}{"sd":>6}{"z":>7}{"p":>8}')
res={}
res['측정 전체']   = run('측정값·전체 53,937', y1, y2)
res['Y2v 전체']    = run('Y2v·전체 53,937',   y1, y2v)
res['측정 평가셋'] = run('측정값·평가셋 5,394', y1[te], y2[te])
res['예측 평가셋'] = run('예측값·평가셋 5,394', p1, p2)
json.dump(res, open(U/'out'/'front_recheck.json','w'), ensure_ascii=False, indent=1)
print('\n[save] front_recheck.json')
