"""
교란 배제 B2 — 상위 구간 고갈이 '나눗셈이 만든 착시'인지 검정
==============================================================
4.5절 첫째·둘째 교란 배제의 산출 코드.
Y2 = HF1/WT 는 분모를 Y1 과 공유한다. 그 공유 자체가 상위 구간 고갈을
만들어낼 수 있으므로, 아래 세 대조군으로 분리한다.
  대조 1  HF1 을 가이드 사이에서 섞어 WT-HF1 관계를 파괴한 뒤에도 고갈이 남는가
  대조 2  분모를 공유하지 않는 정의(HF1 원값, WT 회귀 잔차)에서도 고갈이 남는가
  대조 3  1.5 클리핑(천장)이 고갈에 기여하는가
출력은 표준출력 표. 시드 20260906.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd, math
from scipy import stats
import _env

D = pd.read_csv(_env.data_dir() / 'deephf_3enzyme_derived.csv', index_col=0)
WT=D['wt'].to_numpy(float); HF=D['hf'].to_numpy(float); N=len(WT)
QS=[0.25,0.15,0.10,0.07,0.05,0.03]
rk=stats.rankdata
def joint(a,b,q,n):
    k=int(round(n*q)); return int(np.sum((a>n-k)&(b>n-k)))

def zprofile(y1,y2,mc=500,seed=20260906,tag=""):
    rs=stats.spearmanr(y1,y2)[0]; R=2*math.sin(math.pi*rs/6)
    r1,r2=rk(y1),rk(y2)
    obs={q:joint(r1,r2,q,N) for q in QS}
    L=np.linalg.cholesky([[1,R],[R,1]]); rng=np.random.default_rng(seed)
    acc={q:np.empty(mc) for q in QS}
    for i in range(mc):
        z=rng.standard_normal((N,2))@L.T
        a1,a2=rk(z[:,0]),rk(z[:,1])
        for q in QS: acc[q][i]=joint(a1,a2,q,N)
    zs={q:(obs[q]-acc[q].mean())/acc[q].std(ddof=1) for q in QS}
    print(f"  {tag:<34} rho_s {rs:+.4f} | " + " ".join(f"{q:.2f}:{zs[q]:+7.2f}" for q in QS))
    return zs

print("q 별 z  (양수 = 기대보다 많다, 음수 = 기대보다 적다)")
print("="*118)
print("[실제 데이터]")
real = zprofile(WT, np.clip(HF/WT,None,1.5), tag="Y2 = HF1/WT (현행 정의)")

print("\n[대조 1] HF1을 가이드 사이에서 무작위로 섞는다.")
print("         -> WT와 HF1 사이의 실제 관계는 완전히 파괴된다.")
print("            그래도 상위 구간 고갈이 나오면, 그것은 나눗셈이 만든 착시다.")
for s in [1,2,3]:
    rng=np.random.default_rng(20260906+s)
    HFp=rng.permutation(HF)
    zprofile(WT, np.clip(HFp/WT,None,1.5), tag=f"Y2 = HF1(섞음)/WT  시드 {s}")

print("\n[대조 2] 분모를 공유하지 않는 정의")
zprofile(WT, HF, tag="Y2 = HF1 원값 (나눗셈 없음)")
res = HF - np.poly1d(np.polyfit(WT,HF,1))(WT)
zprofile(WT, res, tag="Y2 = HF1의 WT 회귀 잔차")

print("\n[대조 3] 천장(1.5 클리핑)의 기여")
zprofile(WT, HF/WT, tag="Y2 = HF1/WT, 클리핑 없음")
