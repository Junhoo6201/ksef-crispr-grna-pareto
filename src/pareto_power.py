"""
파레토 전선 크기 검정의 검정력 — 평가셋 규모(n=5,394)에서
=============================================================
상위 5% 동시초과가 z = -3.41 로 상충을 잡아내는 강도에서,
비지배 전선 크기는 귀무 대비 어느 정도 z 를 내는가. 즉 전선 크기
통계량이 그 강도의 상충을 탐지할 검정력이 있는지 본다.
보조 분석 — 10쪽판 원고에 포함되지 않는다. 시드 20260906.
"""
import numpy as np, math

N, MC, SEED = 5394, 2000, 20260906
RHO_S = 0.0324                      # 평가셋 측정 Spearman
R = 2*math.sin(math.pi*RHO_S/6)     # -> Pearson for Gaussian copula

def front_size(x, y):
    """두 목적 모두 최대화. 정렬+스위핑 O(n log n)."""
    o = np.argsort(-x, kind='mergesort')
    ys = y[o]; best = -np.inf; c = 0
    for v in ys:
        if v > best:
            c += 1; best = v
    return c

def joint_top(x, y, q):
    k = int(round(len(x)*q))
    tx = np.argsort(-x)[:k]; ty = np.argsort(-y)[:k]
    return len(np.intersect1d(tx, ty))

def draw(rng, a=0.0, s0=1.0, b=0.5):
    """a=0 이면 귀무(가우시안 코퓰러). a>0 이면 상위 구간에서만
       충돌 방향(d) 분산을 키워 '꼬리 한정 상충'을 만든다."""
    z = rng.multivariate_normal([0,0], [[1,R],[R,1]], size=N)
    z1, z2 = z[:,0], z[:,1]
    if a > 0:
        s = (z1+z2)/math.sqrt(2); d = (z1-z2)/math.sqrt(2)
        d = d*(1 + a/(1+np.exp(-(s-s0)/b)))
        z1, z2 = (s+d)/math.sqrt(2), (s-d)/math.sqrt(2)
    return z1, z2

# ---------- 1. 귀무분포 재현 확인 ----------
rng = np.random.default_rng(SEED)
null_f = np.array([front_size(*draw(rng)) for _ in range(MC)])
print(f"귀무 전선 크기 : {null_f.mean():.2f} +- {null_f.std(ddof=1):.2f}   (기록값 8.83 +- 2.72)")
print(f"독립 이론값 H_n: {sum(1/k for k in range(1,N+1)):.2f}")

# ---------- 2. 대립가설 강도 보정: 상위 5% 동시초과 z = -3.41 ----------
rng = np.random.default_rng(SEED)
null_j5 = np.array([joint_top(*draw(rng), 0.05) for _ in range(500)])
m5, s5 = null_j5.mean(), null_j5.std(ddof=1)
print(f"\n귀무 상위5% 동시초과: {m5:.2f} +- {s5:.2f}   (기록값 15.48 +- 3.66)")

print("\n a     상위5% z    상위25% z   전선크기(평균+-표준편차)   전선 z")
best = None
for a in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    rng = np.random.default_rng(SEED+1)
    js, j25, fs = [], [], []
    for _ in range(300):
        x, y = draw(rng, a=a)
        js.append(joint_top(x, y, 0.05)); j25.append(joint_top(x, y, 0.25))
        fs.append(front_size(x, y))
    js, j25, fs = map(np.array, (js, j25, fs))
    rng2 = np.random.default_rng(SEED)
    n25 = np.array([joint_top(*draw(rng2), 0.25) for _ in range(300)])
    z5 = (js.mean()-m5)/s5
    z25 = (j25.mean()-n25.mean())/n25.std(ddof=1)
    zf = (fs.mean()-null_f.mean())/null_f.std(ddof=1)
    print(f"{a:4.1f}   {z5:+7.2f}    {z25:+7.2f}      {fs.mean():6.2f} +- {fs.std(ddof=1):4.2f}      {zf:+6.2f}")
    if best is None or abs(z5-(-3.41)) < abs(best[1]-(-3.41)):
        best = (a, z5, fs.mean(), fs.std(ddof=1), zf)

a, z5, fm, fsd, zf = best
print(f"\n관측 z(-3.41)에 가장 가까운 강도 a = {a}  (상위5% z = {z5:+.2f})")
print(f"  그 대립가설에서 전선 크기 = {fm:.2f} +- {fsd:.2f}  -> 귀무 대비 z = {zf:+.2f}")
print(f"  실제 관측 전선 = 11 (z = +0.80)")
