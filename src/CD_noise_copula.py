"""
교란 배제 C·D — 측정 잡음과 귀무 코퓰러 족
============================================
4.5절 셋째·넷째 교란 배제의 산출 코드.
  PART C  잠재값에 꼬리 상충이 전혀 없는 가우시안을 두고, 리드 수 기반
          이항 잡음만 더한다. 그래도 부호 전환이 나오면 '승자의 저주'
          (측정 잡음)로 설명되므로 C 가 교란 후보로 남고, 안 나오면 배제된다.
  PART D  귀무 코퓰러 족을 가우시안 → t(nu=3,5,10) → Clayton → Gumbel 로
          바꿔도 상위 구간 고갈이 남는지 본다. 족 교체에 견디면 결과가
          가우시안 가정의 산물이 아니다.
출력은 표준출력 표. 시드 20260906.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd, math
from scipy import stats
import _env

D = pd.read_csv(_env.data_dir() / 'deephf_3enzyme_derived.csv', index_col=0)
WT=D['wt'].to_numpy(float); HF=D['hf'].to_numpy(float); N=len(WT)
QS=[0.25,0.15,0.12,0.10,0.07,0.05,0.03]
rk=stats.rankdata
def joint(a,b,q,n):
    k=int(round(n*q)); return int(np.sum((a>n-k)&(b>n-k)))
def obs_counts(y1,y2):
    r1,r2=rk(y1),rk(y2); return {q:joint(r1,r2,q,N) for q in QS}
def zline(tag,obs,nullacc,extra=""):
    out=[]
    for q in QS:
        m,s=nullacc[q].mean(),nullacc[q].std(ddof=1)
        out.append(f"{q:.2f}:{(obs[q]-m)/s:+7.2f}")
    print(f"  {tag:<40}{extra} " + " ".join(out))

# ============ PART C : 이항(리드) 잡음이 부호 전환을 만들 수 있는가 ============
print("="*126)
print("PART C — 승자의 저주 / 측정 잡음")
print("  설계: 잠재값은 가우시안(= 꼬리 상충 없음). 거기에 리드 수 기반 이항 잡음만 더한다.")
print("        부호 전환이 나오면 C가 설명이 되고, 안 나오면 C는 배제된다.")
WTs=np.sort(WT); HFs=np.sort(HF)
def simC(rho, nreads, seed):
    rng=np.random.default_rng(seed)
    L=np.linalg.cholesky([[1,rho],[rho,1]]); z=rng.standard_normal((N,2))@L.T
    p1=WTs[np.clip((stats.rankdata(z[:,0])-1).astype(int),0,N-1)]
    p2=HFs[np.clip((stats.rankdata(z[:,1])-1).astype(int),0,N-1)]
    o1=rng.binomial(nreads,np.clip(p1,0,1))/nreads
    o2=rng.binomial(nreads,np.clip(p2,0,1))/nreads
    o1=o1+rng.normal(0,1e-9,N); o2=o2+rng.normal(0,1e-9,N)   # 동점 무작위 분리
    return o1,o2
for nreads in [100,1000]:
    lo,hi=0.0,0.999
    for _ in range(18):                       # 관측 rho_s = +0.7264 에 맞춤
        mid=(lo+hi)/2
        if stats.spearmanr(*simC(mid,nreads,1))[0] < 0.7264: lo=mid
        else: hi=mid
    rho=(lo+hi)/2
    o1,o2=simC(rho,nreads,7); rs=stats.spearmanr(o1,o2)[0]
    R=2*math.sin(math.pi*rs/6); Lc=np.linalg.cholesky([[1,R],[R,1]])
    rng=np.random.default_rng(20260906); acc={q:np.empty(300) for q in QS}
    for i in range(300):
        z=rng.standard_normal((N,2))@Lc.T; a1,a2=rk(z[:,0]),rk(z[:,1])
        for q in QS: acc[q][i]=joint(a1,a2,q,N)
    zline(f"잠재 가우시안 + 이항잡음 (리드 {nreads})", obs_counts(o1,o2), acc, f" rho_s {rs:+.4f}")
print("  " + "-"*120)
# 실제 데이터 (나눗셈 없는 정의, 같은 축에서 비교)
rs=stats.spearmanr(WT,HF)[0]; R=2*math.sin(math.pi*rs/6)
Lc=np.linalg.cholesky([[1,R],[R,1]]); rng=np.random.default_rng(20260906)
accR={q:np.empty(300) for q in QS}
for i in range(300):
    z=rng.standard_normal((N,2))@Lc.T; a1,a2=rk(z[:,0]),rk(z[:,1])
    for q in QS: accR[q][i]=joint(a1,a2,q,N)
obsR=obs_counts(WT,HF)
zline("실제 데이터 (Y2 = HF1 원값)", obsR, accR, f" rho_s {rs:+.4f}")
print(f"\n  과제 3 — 교차점: " + " ".join(f"q={q:.2f}:{obsR[q]}/{accR[q].mean():.0f}" for q in [0.15,0.12,0.10]))

# ============ PART D : 귀무 코퓰러 족을 바꾸면 결과가 사라지는가 ============
print("\n"+"="*126)
print("PART D — 귀무 코퓰러 족 교체 (현행 정의 Y2 = HF1/WT, rho_s = +0.0253)")
Y2=np.clip(HF/WT,None,1.5); TGT=stats.spearmanr(WT,Y2)[0]
obsM=obs_counts(WT,Y2)

def samp_gauss(rng,th):
    L=np.linalg.cholesky([[1,th],[th,1]]); z=rng.standard_normal((N,2))@L.T; return z[:,0],z[:,1]
def samp_t(nu):
    def f(rng,th):
        L=np.linalg.cholesky([[1,th],[th,1]]); z=rng.standard_normal((N,2))@L.T
        w=rng.chisquare(nu,N)/nu; w=np.sqrt(w)[:,None]
        t=z/w; return t[:,0],t[:,1]
    return f
def samp_clayton(rng,th):
    th=max(th,1e-6); v1=rng.random(N); v2=rng.random(N)
    u2=(v1**(-th)*(v2**(-th/(1+th))-1)+1)**(-1/th); return v1,u2
def samp_gumbel(rng,th):
    th=max(th,1.0+1e-9); al=1/th
    U=rng.random(N)*math.pi; W=rng.exponential(1,N)
    S=(np.sin(al*U)/np.sin(U)**(1/al))*(np.sin((1-al)*U)/W)**((1-al)/al)
    E1,E2=rng.exponential(1,N),rng.exponential(1,N)
    return np.exp(-(E1/S)**al), np.exp(-(E2/S)**al)

def tune(f,lo,hi):
    for _ in range(22):
        mid=(lo+hi)/2
        r=stats.spearmanr(*f(np.random.default_rng(3),mid))[0]
        if r<TGT: lo=mid
        else: hi=mid
    return (lo+hi)/2

for name,f,lo,hi in [("가우시안 (현행)",samp_gauss,-0.5,0.5),
                     ("t-코퓰러 nu=3  (양쪽 꼬리 의존)",samp_t(3),-0.5,0.5),
                     ("t-코퓰러 nu=5",samp_t(5),-0.5,0.5),
                     ("t-코퓰러 nu=10",samp_t(10),-0.5,0.5),
                     ("Clayton (아래 꼬리 의존)",samp_clayton,1e-6,2.0),
                     ("Gumbel  (위 꼬리 의존)",samp_gumbel,1.0,2.0)]:
    th=tune(f,lo,hi); rng=np.random.default_rng(20260906)
    acc={q:np.empty(300) for q in QS}
    for i in range(300):
        x,y=f(rng,th); a1,a2=rk(x),rk(y)
        for q in QS: acc[q][i]=joint(a1,a2,q,N)
    rchk=stats.spearmanr(*f(np.random.default_rng(99),th))[0]
    zline(name, obsM, acc, f" rho_s {rchk:+.4f}")
print(f"\n  독립 귀무 N*q^2 (족과 무관): " + " ".join(f"{q:.2f}:{N*q*q:.0f}" for q in QS))
print(f"  실제 관측                   : " + " ".join(f"{q:.2f}:{obsM[q]}" for q in QS))
