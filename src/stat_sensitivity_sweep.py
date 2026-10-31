"""
통계량 민감도 스윕 — 꼬리 상충 강도 a 에 따른 세 통계량의 반응
================================================================
같은 자료에 대해 상위 분위 동시초과 수, 비지배 전선 크기, 1-HV 가
서로 다른 민감도를 갖는지 본다. 꼬리 한정 상충을 강도 a 로 주입한
합성 자료에서 각 통계량의 z 를 비교한다.
보조 분석 — 10쪽판 원고에 포함되지 않는다. 시드 20260906/20260907.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd, math
from scipy import stats
import _env

D = pd.read_csv(_env.data_dir() / 'deephf_3enzyme_derived.csv', index_col=0)
Y1=D['wt'].to_numpy(float); Y2=np.clip(D['ratio_hf'].to_numpy(float),None,1.5)
N=len(Y1); RS=stats.spearmanr(Y1,Y2)[0]; R=2*math.sin(math.pi*RS/6); SQ2=math.sqrt(2.)
ranks=stats.rankdata
def joint(r1,r2,q,n):
    k=int(round(n*q)); return int(np.sum((r1>n-k)&(r2>n-k)))
def front_hv(x,y,n):
    u=ranks(x)/n; w=ranks(y)/n
    o=np.argsort(-u,kind='mergesort'); us=u[o]; ws=w[o]
    run=np.maximum.accumulate(ws)
    keep=np.concatenate(([True],ws[1:]>run[:-1]))
    fu=us[keep]; fw=ws[keep]
    nxt=np.concatenate((fu[1:],[0.0]))
    return int(keep.sum()), float(np.sum((fu-nxt)*fw))

r1,r2=ranks(Y1),ranks(Y2); OF,OH=front_hv(Y1,Y2,N)
print(f"관측: 전선 {OF}  HV {OH:.6f}  (1-HV = {1-OH:.6f})  상위5% {joint(r1,r2,0.05,N)}")
print(f"독립 이론 H_n = {sum(1/k for k in range(1,N+1)):.2f}\n")

def gen(rng,r0,a,s_top=2.3,b=0.2):
    L=np.linalg.cholesky([[1,r0],[r0,1]]); z=rng.standard_normal((N,2))@L.T
    s=(z[:,0]+z[:,1])/SQ2; d=(z[:,0]-z[:,1])/SQ2
    d=d*(1+a/(1+np.exp(-(s-s_top)/b)))
    return (s+d)/SQ2,(s-d)/SQ2

def run(a,reps,seed=20260906):
    rng=np.random.default_rng(seed); F=[];H=[];J5=[];J25=[]
    for _ in range(reps):
        x,y=gen(rng,R,a); f,h=front_hv(x,y,N); F.append(f); H.append(h)
        b1,b2=ranks(x),ranks(y)
        J5.append(joint(b1,b2,0.05,N)); J25.append(joint(b1,b2,0.25,N))
    return map(np.array,(F,H,J5,J25))

print("=== 귀무분포 (a=0, 1,000회, 시드 20260906) ===")
F0,H0,J50,J250=run(0.0,1000)
print(f"  전선      {F0.mean():8.2f} ± {F0.std(ddof=1):.3f}   관측 {OF}  z = {(OF-F0.mean())/F0.std(ddof=1):+.2f}")
print(f"  1-HV      {(1-H0).mean():8.6f} ± {(1-H0).std(ddof=1):.6f}   관측 {1-OH:.6f}  z = {((1-OH)-(1-H0).mean())/(1-H0).std(ddof=1):+.2f}")
print(f"  상위5%    {J50.mean():8.2f} ± {J50.std(ddof=1):.2f}   관측 27  z = {(27-J50.mean())/J50.std(ddof=1):+.2f}")

print("\n=== 과제 3: 꼬리 충돌 강도 a 에 따른 각 통계량의 z (각 200회) ===")
print(f"{'a':>5}{'상위25% 관측':>13}{'상위5% z':>11}{'전선 z':>10}{'1-HV z':>11}")
s0f,s0h,s0j=F0.std(ddof=1),(1-H0).std(ddof=1),J50.std(ddof=1)
m0f,m0h,m0j=F0.mean(),(1-H0).mean(),J50.mean()
for a in [0.0,0.25,0.5,1.0,2.0,4.0]:
    F,H,J5,J25=run(a,200,seed=20260907)
    print(f"{a:5.2f}{J25.mean():13.0f}{(J5.mean()-m0j)/s0j:+11.2f}"
          f"{(F.mean()-m0f)/s0f:+10.2f}{((1-H).mean()-m0h)/s0h:+11.2f}")
print(f"\n(참고) 실제 관측  상위25% 4261   상위5% z -10.65   전선 z +3.19   1-HV z 위 참조")
