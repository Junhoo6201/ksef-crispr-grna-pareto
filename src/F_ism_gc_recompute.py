"""2단계 — (A) 발표 수치 재현 게이트  (B) GC 3종 재계산 ISM
원고 4.8절·표 4 의 ΔŶ 가 서열 경로에 한정된 기여분임을 확인하고,
서열에서 정확히 재계산 가능한 GC 3종을 함께 갱신했을 때 판정이 바뀌는지 본다.
"""
import numpy as np, torch, torch.nn as nn, pathlib, json, sys
U = pathlib.Path(__file__).resolve().parents[1]
NTS = list("ACGT")
BIO = ['ddg_1','dg_open','dg_bind20','dg_bind7to20','gc_gt10','gc_lt10','gc_count',
       'tm_global','tm2','tm3','dg_stem']
I_GT, I_LT, I_CNT = BIO.index('gc_gt10'), BIO.index('gc_lt10'), BIO.index('gc_count')

class MultiTaskCNN(nn.Module):                     # phase2_model.py 와 동일
    def __init__(self, n_bio=11, ch=128, emb=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv1d(4, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.merge = nn.Sequential(
            nn.Linear(ch + n_bio, emb*2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(emb*2, emb), nn.ReLU())
        head = lambda: nn.Sequential(nn.Linear(emb,32), nn.ReLU(), nn.Linear(32,1))
        self.head_eff, self.head_fid = head(), head()
    def forward(self, x_seq, x_bio):
        h = self.pool(self.trunk(x_seq)).squeeze(-1)
        z = self.merge(torch.cat([h, x_bio], 1))
        return self.head_eff(z).squeeze(-1), self.head_fid(z).squeeze(-1)

torch.set_num_threads(2)
D = dict(np.load(U/'out/dataset.npz', allow_pickle=True))
m = MultiTaskCNN(D['X_bio'].shape[1])
m.load_state_dict(torch.load(U/'out/model_lam050.pt', map_location='cpu'))
m.eval()
te = D['split'] == 2
xs, xb, seq = D['X_seq'][te], D['X_bio'][te], D['seq'][te]
mu, sd = D['bio_mu'], D['bio_sd']
N = len(xs)
print(f'평가셋 {N:,}개  |  torch {torch.__version__}')

def fwd(xs_, xb_, bs=1024):
    o1,o2=[],[]
    with torch.no_grad():
        for i in range(0,len(xs_),bs):
            a,b = m(torch.tensor(xs_[i:i+bs]), torch.tensor(xb_[i:i+bs]))
            o1.append(a.numpy()); o2.append(b.numpy())
    return np.concatenate(o1), np.concatenate(o2)

b1, b2 = fwd(xs, xb)
# 원값 복원 → GC 재계산용
raw = xb*sd + mu
gc0 = np.array([s.count('G')+s.count('C') for s in seq], np.float32)
assert np.allclose(raw[:,I_CNT], gc0, atol=1e-3), 'gc_count 복원 불일치'
GC_SET = set('GC')
orig_nt = xs.argmax(1)                               # (N,20)

def ism_pair(pos, nt, fix_gc):
    """해당 위치에 그 염기가 '아닌' 가이드만 대상으로 ΔŶ 평균."""
    mask = orig_nt[:,pos] != nt
    if mask.sum() == 0: return None
    xm = xs[mask].copy(); xm[:,:,pos]=0.0; xm[:,nt,pos]=1.0
    xbm = xb[mask].copy()
    if fix_gc:
        old_gc_is = np.isin(orig_nt[mask,pos], [NTS.index('G'), NTS.index('C')])
        new_gc_is = NTS[nt] in GC_SET
        g = gc0[mask] + (1.0 if new_gc_is else 0.0) - old_gc_is.astype(np.float32)
        r = raw[mask].copy()
        r[:,I_CNT] = g; r[:,I_GT] = (g>10); r[:,I_LT] = (g<10)
        xbm = ((r-mu)/sd).astype(np.float32)
    q1,q2 = fwd(xm, xbm)
    return dict(n=int(mask.sum()),
                dY1=float((q1-b1[mask]).mean()), dY2=float((q2-b2[mask]).mean()))

# ---------- (A) 표 4 재현 게이트 : 위치 18 G 보유 가이드 ----------
pos18, iG = 17, NTS.index('G')
has = orig_nt[:,pos18] == iG
print(f'\n[게이트] 위치 18 에 G 를 가진 평가셋 가이드 n = {int(has.sum()):,}  (원고 1,199)')
PUB = {'A':(+0.108,-0.096), 'T':(+0.081,-0.131), 'C':(+0.088,+0.010)}
print(f'{"치환":>6}{"ΔŶ₂ 원본코드":>13}{"원고":>9}{"ΔŶ₁ 원본코드":>14}{"원고":>9} | {"ΔŶ₂ GC재계산":>13}{"ΔŶ₁ GC재계산":>14}')
rows={}
for alt in range(4):
    if alt == iG: continue
    out={}
    for fix in (False, True):
        xm = xs[has].copy(); xm[:,:,pos18]=0.0; xm[:,alt,pos18]=1.0
        xbm = xb[has].copy()
        if fix:
            g = gc0[has] + (1.0 if NTS[alt] in GC_SET else 0.0) - 1.0   # 원래가 G 이므로 -1
            r = raw[has].copy(); r[:,I_CNT]=g; r[:,I_GT]=(g>10); r[:,I_LT]=(g<10)
            xbm = ((r-mu)/sd).astype(np.float32)
        q1,q2 = fwd(xm, xbm)
        out[fix] = (float((q2-b2[has]).mean()), float((q1-b1[has]).mean()))
    p2,p1 = PUB[NTS[alt]]
    rows[NTS[alt]] = out
    print(f'  G→{NTS[alt]} {out[False][0]:+12.4f}{p2:+9.3f}{out[False][1]:+14.4f}{p1:+9.3f} | '
          f'{out[True][0]:+13.4f}{out[True][1]:+14.4f}')

# ---------- (B) 상위/하위 10 치환 — 두 방식 비교 ----------
print('\n[B] 20위치 × 4염기 전수 — 원본 코드 기준 ΔŶ₂ 하위5·상위5 와 GC 재계산 대조')
cand=[]
for pos in range(20):
    for nt in range(4):
        r = ism_pair(pos, nt, False)
        if r is None or r['n'] < 50: continue
        g = ism_pair(pos, nt, True)
        cand.append(dict(pos=pos+1, nt=NTS[nt], n=r['n'],
                         dY2=r['dY2'], dY1=r['dY1'], dY2g=g['dY2'], dY1g=g['dY1']))
cand.sort(key=lambda c:c['dY2'])
top10 = cand[:5] + cand[-5:][::-1]
print(f'{"치환":>8}{"n":>7}{"ΔŶ₂":>9}{"ΔŶ₁":>9}{"부호반대":>9} | {"ΔŶ₂(GC)":>9}{"ΔŶ₁(GC)":>9}{"부호반대":>9}')
opp_o=opp_g=0
for c in top10:
    o = (c['dY2']>0) != (c['dY1']>0); gg = (c['dY2g']>0) != (c['dY1g']>0)
    opp_o += o; opp_g += gg
    print(f'  pos{c["pos"]:2d}→{c["nt"]}{c["n"]:7,}{c["dY2"]:+9.4f}{c["dY1"]:+9.4f}{"O" if o else "X":>9} | '
          f'{c["dY2g"]:+9.4f}{c["dY1g"]:+9.4f}{"O" if gg else "X":>9}')
print(f'\n반대 방향: 원본 코드 {opp_o}/10  ·  GC 재계산 {opp_g}/10   (원고 서술: 8/10)')
json.dump(dict(gate_pos18=rows, top10=top10, opp_orig=int(opp_o), opp_gc=int(opp_g)),
          open(U/'out'/'ism_gc_recompute.json','w'), indent=1)
print('[save] out/ism_gc_recompute.json')
