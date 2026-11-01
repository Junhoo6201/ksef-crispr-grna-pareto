"""ISM 재계산 범위 확장 — GC 3종 + 융해온도 3종 (11개 중 6개)
tm_global = Tm_NN(seq21[0:21]) · tm2 = Tm_NN(seq21[15:21]) · tm3 = Tm_NN(seq21[4:13])
(biopython 기본값, 55,604행 전수 일치 확인)"""
import numpy as np, torch, torch.nn as nn, pandas as pd, pathlib, json
from Bio.SeqUtils import MeltingTemp as mt
U = pathlib.Path(__file__).resolve().parents[1]
NTS=list("ACGT")
BIO=['ddg_1','dg_open','dg_bind20','dg_bind7to20','gc_gt10','gc_lt10','gc_count','tm_global','tm2','tm3','dg_stem']
I={k:BIO.index(k) for k in ('gc_gt10','gc_lt10','gc_count','tm_global','tm2','tm3')}

class MultiTaskCNN(nn.Module):
    def __init__(self,n_bio=11,ch=128,emb=64):
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

torch.set_num_threads(4)
D=dict(np.load(U/'out'/'dataset.npz',allow_pickle=True))
te=D['split']==2
xs,xb,seq = D['X_seq'][te], D['X_bio'][te], D['seq'][te]
mu,sd = D['bio_mu'], D['bio_sd']; raw = xb*sd+mu
dec=pd.read_csv(U/'data'/'deephf_wt_decoded.csv', usecols=['seq20','pos21']).set_index('seq20')['pos21']
pos21=np.array([dec.loc[s] for s in seq]); assert len(pos21)==len(seq)
m=MultiTaskCNN(xb.shape[1]); m.load_state_dict(torch.load(U/'out'/'model_lam050.pt',map_location='cpu')); m.eval()
def fwd(a,b,bs=2048):
    o1,o2=[],[]
    with torch.no_grad():
        for i in range(0,len(a),bs):
            x,y=m(torch.tensor(a[i:i+bs]),torch.tensor(b[i:i+bs])); o1.append(x.numpy()); o2.append(y.numpy())
    return np.concatenate(o1),np.concatenate(o2)
b1,b2=fwd(xs,xb)
orig=xs.argmax(1)
# 치환 서열 문자열 생성용
seq_arr=np.array([list(s) for s in seq])
TMC={}   # 캐시
def tmv(s):
    v=TMC.get(s)
    if v is None:
        v=(mt.Tm_NN(s[0:21]), mt.Tm_NN(s[15:21]), mt.Tm_NN(s[4:13])); TMC[s]=v
    return v

def xb_for(mask, pos, nt, mode):
    """mode: 'orig'(원 구현) | 'gc' | 'gc_tm'"""
    if mode=='orig': return xb[mask].copy()
    r=raw[mask].copy()
    sub=seq_arr[mask].copy(); sub[:,pos]=NTS[nt]
    s21=[''.join(row)+p for row,p in zip(sub, pos21[mask])]
    g=np.array([s[:20].count('G')+s[:20].count('C') for s in s21],float)
    r[:,I['gc_count']]=g; r[:,I['gc_gt10']]=(g>10); r[:,I['gc_lt10']]=(g<10)
    if mode=='gc_tm':
        t=np.array([tmv(s) for s in s21])
        r[:,I['tm_global']]=t[:,0]; r[:,I['tm2']]=t[:,1]; r[:,I['tm3']]=t[:,2]
    return ((r-mu)/sd).astype(np.float32)

def eff(pos,nt,mode,mask):
    xm=xs[mask].copy(); xm[:,:,pos]=0.0; xm[:,nt,pos]=1.0
    q1,q2=fwd(xm, xb_for(mask,pos,nt,mode))
    return float((q2-b2[mask]).mean()), float((q1-b1[mask]).mean())

# --- 표 4 : 위치 18 G 보유
pos18, iG = 17, NTS.index('G'); has = orig[:,pos18]==iG
print(f'[표 4] 위치 18 G 보유 n={int(has.sum()):,}')
print(f'{"치환":>6}{"ΔŶ₂ 원구현":>11}{"ΔŶ₂ GC":>9}{"ΔŶ₂ GC+Tm":>11} | {"ΔŶ₁ 원구현":>11}{"ΔŶ₁ GC":>9}{"ΔŶ₁ GC+Tm":>11}')
t4={}
for alt in range(4):
    if alt==iG: continue
    v={md: eff(pos18,alt,md,has) for md in ('orig','gc','gc_tm')}
    t4[NTS[alt]]=v
    print(f'  G→{NTS[alt]}{v["orig"][0]:+11.4f}{v["gc"][0]:+9.4f}{v["gc_tm"][0]:+11.4f} | '
          f'{v["orig"][1]:+11.4f}{v["gc"][1]:+9.4f}{v["gc_tm"][1]:+11.4f}')

# --- 상위/하위 10 치환
cand=[]
for pos in range(20):
    for nt in range(4):
        msk = orig[:,pos]!=nt
        if msk.sum()<50: continue
        o=eff(pos,nt,'orig',msk)
        cand.append(dict(pos=pos+1,nt=NTS[nt],n=int(msk.sum()),dY2=o[0],dY1=o[1],mask=msk))
cand.sort(key=lambda c:c['dY2'])
top=cand[:5]+cand[-5:][::-1]
print(f'\n{"치환":>8}{"n":>7}{"ΔŶ₂":>9}{"ΔŶ₁":>9}{"반대":>5} | {"ΔŶ₂gc":>8}{"ΔŶ₁gc":>8}{"반대":>5} | {"ΔŶ₂+Tm":>9}{"ΔŶ₁+Tm":>9}{"반대":>5}')
cnt={'orig':0,'gc':0,'gc_tm':0}; rows=[]
for c in top:
    g=eff(c['pos']-1,NTS.index(c['nt']),'gc',c['mask'])
    gt=eff(c['pos']-1,NTS.index(c['nt']),'gc_tm',c['mask'])
    o_=( (c['dY2']>0)!=(c['dY1']>0) ); g_=((g[0]>0)!=(g[1]>0)); t_=((gt[0]>0)!=(gt[1]>0))
    cnt['orig']+=o_; cnt['gc']+=g_; cnt['gc_tm']+=t_
    rows.append(dict(pos=c['pos'],nt=c['nt'],n=c['n'],orig=[c['dY2'],c['dY1']],gc=list(g),gc_tm=list(gt)))
    print(f'  pos{c["pos"]:2d}→{c["nt"]}{c["n"]:7,}{c["dY2"]:+9.4f}{c["dY1"]:+9.4f}{"O" if o_ else "X":>5} | '
          f'{g[0]:+8.4f}{g[1]:+8.4f}{"O" if g_ else "X":>5} | {gt[0]:+9.4f}{gt[1]:+9.4f}{"O" if t_ else "X":>5}')
print(f'\n반대 방향 — 원 구현 {cnt["orig"]}/10 · GC 3종 {cnt["gc"]}/10 · GC+Tm 6종 {cnt["gc_tm"]}/10')
json.dump(dict(table4=t4, top10=rows, opposite=cnt), open(U/'out'/'ism_tm_recompute.json','w'), ensure_ascii=False, indent=1)
print('[save] ism_tm_recompute.json')
