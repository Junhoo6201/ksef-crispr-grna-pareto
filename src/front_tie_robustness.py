# claude/data/front_tie_robustness.py 재실행 (CSV 경로만 조정)
import numpy as np, pandas as pd, math, json
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _env
ROOT = pathlib.Path(__file__).resolve().parent.parent
D=pd.read_csv(_env.data_dir() / 'deephf_3enzyme_derived.csv',index_col=0)
Y1=D['wt'].to_numpy(float); Y2=np.clip(D['ratio_hf'].to_numpy(float),None,1.5)
N=len(Y1)
print("=== 동점 구조 ===")
for nm,v in [("Y1",Y1),("Y2",Y2)]:
    u,c=np.unique(v,return_counts=True)
    print(f"  {nm}: 고유값 {len(u):,}  동점에 묶인 관측 {int(c[c>1].sum()):,}  최대 동점군 {c.max()}")
print(f"  Y2 상한 1.5 동점 {int((Y2>=1.5).sum())}")
def front_hv(u,w):
    o=np.argsort(-u,kind='mergesort'); us=u[o]; ws=w[o]
    run=np.maximum.accumulate(ws)
    keep=np.concatenate(([True],ws[1:]>run[:-1]))
    fu=us[keep]; fw=ws[keep]; nxt=np.concatenate((fu[1:],[0.0]))
    return int(keep.sum()), float(np.sum((fu-nxt)*fw))
a1=stats.rankdata(Y1)/N; a2=stats.rankdata(Y2)/N
f_avg,h_avg=front_hv(a1,a2)
rng=np.random.default_rng(20260906); FS=[];HS=[]
for _ in range(30):
    j1=stats.rankdata(Y1+rng.normal(0,1e-12,N))/N
    j2=stats.rankdata(Y2+rng.normal(0,1e-12,N))/N
    f,h=front_hv(j1,j2); FS.append(f); HS.append(h)
FS=np.array(FS); HS=np.array(HS)
print(f"\n=== 동점 처리에 따른 관측값 ===")
print(f"  평균순위(현행)     전선 {f_avg}   1-HV {1-h_avg:.6f}")
print(f"  무작위 분리 30회   전선 {FS.min()}–{FS.max()} (평균 {FS.mean():.1f})   1-HV {(1-HS).min():.6f}–{(1-HS).max():.6f}")
RS=stats.spearmanr(Y1,Y2)[0]; R=2*math.sin(math.pi*RS/6)
L=np.linalg.cholesky([[1,R],[R,1]]); rng=np.random.default_rng(20260906)
F0=[];H0=[]
for _ in range(2000):
    z=rng.standard_normal((N,2))@L.T
    f,h=front_hv(stats.rankdata(z[:,0])/N, stats.rankdata(z[:,1])/N); F0.append(f); H0.append(h)
F0=np.array(F0); H0=1-np.array(H0)
print(f"\n=== 귀무 (2,000회) ===")
print(f"  전선  평균 {F0.mean():.2f} ± {F0.std(ddof=1):.2f}   최대 {F0.max()}   95%분위 {np.percentile(F0,95):.0f}")
for lab,v in [("평균순위",f_avg),("무작위 최소",FS.min()),("무작위 최대",FS.max())]:
    p=(F0>=v).mean()
    print(f"    관측 {v:>3} ({lab:8}) -> z {(v-F0.mean())/F0.std(ddof=1):+.2f}, p = {p:.4f} ({int((F0>=v).sum())}/2000)")
# 독립 이론 Hn (조화수)
Hn = sum(1.0/i for i in range(1, N+1))
print(f"  독립 이론 Hₙ = {Hn:.2f}")
json.dump(dict(n=N, seed=20260906, reps=2000,
               observed_avgrank=int(f_avg), tie_range=[int(FS.min()),int(FS.max())],
               null_mean=float(F0.mean()), null_sd=float(F0.std(ddof=1)),
               null_counts=np.bincount(F0, minlength=40).tolist(),
               p_avgrank=float((F0>=f_avg).mean()), p_tie_min=float((F0>=FS.min()).mean()),
               independent_Hn=float(Hn)),
          open(ROOT / 'out/front_fullsample.json','w'), ensure_ascii=False, indent=1)
print("saved out/front_fullsample.json")
