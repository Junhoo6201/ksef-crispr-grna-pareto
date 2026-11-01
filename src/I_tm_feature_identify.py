"""DeepHF 융해온도 3종의 계산식 복원 — 전수 대조 (2026-11-01)
결과: tm_global = Tm_NN(seq21[0:21]) · tm2 = Tm_NN(seq21[15:21]) · tm3 = Tm_NN(seq21[4:13])
      biopython MeltingTemp.Tm_NN 기본 인자, 55,604행 전부 일치(최대오차 7.1e-15).
자유에너지 5종(ddg_1·dg_open·dg_bind20·dg_bind7to20·dg_stem)은 원 논문이
RNAfold 2.4.5 사용만 밝히고 특징별 정의를 적지 않아 복원하지 못하였다.
"""
import pandas as pd, numpy as np, pathlib, Bio
from Bio.SeqUtils import MeltingTemp as mt
ROOT = pathlib.Path(__file__).resolve().parents[1]
D = pd.read_csv(ROOT/'data'/'deephf_wt_decoded.csv')
print(f'biopython {Bio.__version__}  rows {len(D):,}')
s21 = D['seq21'].to_numpy()
SPEC = {'tm_global': (0, 21), 'tm2': (15, 21), 'tm3': (4, 13)}
ok = True
for nm, (a, b) in SPEC.items():
    calc = np.array([mt.Tm_NN(s[a:b]) for s in s21])
    d = np.abs(calc - D[nm].to_numpy(float))
    hit = int((d < 1e-9).sum())
    ok &= hit == len(d)
    print(f'{nm:10s} = Tm_NN(seq21[{a}:{b}])  최대오차 {d.max():.3e}  일치 {hit:,}/{len(d):,}')
print('[결과]', '전수 일치 — 계산식 복원 성공' if ok else '불일치 — 복원 실패')
# 탐색 과정 재현: 모든 연속 부분열에 적용해 일치 구간을 찾는다
S = D.head(6).to_dict('records')
print('\n[탐색 재현] 앞 6행으로 (시작, 끝) 전수 탐색')
for nm in SPEC:
    tv = [r[nm] for r in S]
    found = [(a, b) for a in range(21) for b in range(a+2, 22)
             if all(abs(mt.Tm_NN(r['seq21'][a:b]) - t) < 1e-6 for r, t in zip(S, tv))]
    print(f'  {nm:10s} → {found}')
