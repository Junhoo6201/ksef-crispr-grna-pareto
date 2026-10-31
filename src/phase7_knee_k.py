"""
Phase 7 — 무릎점 선별의 k 민감도
================================
주의: 이 분석과 그 바탕인 선별 전략 비교(V4)는 10쪽판 원고에 포함되지 않는다.
      제외 사유는 분량이 아니다. V4 는 예측값으로 가이드를 고르므로 재학습하면
      결과가 무너진다 — 가중합 − 효율 단독의 측정 Y₂ 이득이 원 실행 +0.098 에서
      재실행 +0.010 으로 약 1/10 이 되었고, 같은 재실행에서 예측 전선 26 → 20,
      예측 기준 동시 만족 587·87·0 → 565·121·7 로 바뀌었다.
      (측정값만으로 계산되는 양은 한 자리도 움직이지 않았다.)
      아래 절 번호는 구판 원고의 것이다.

문제 제기: 구판 4.3절의 선별 전략 비교는 k = 20으로 고정되어 있다. 그런데 예측
파레토 전선 F₁의 크기는 26개이므로, k = 20은 전선의 77%를 그대로 가져오는
것이다. 이 경우 "무릎점 선별"은 무릎 구간이 아니라 전선 전체(저효율 꼬리 포함)를
고르는 것과 같아지고, 낮은 측정 Y₁은 전략의 성질이 아니라 k 설정의 결과일 수 있다.

k ∈ {5, 10, 20, 26} 에서 네 전략을 동일 조건으로 재비교한다.
평가는 모두 '측정값'으로 한다. 예측값으로 고르고 측정값으로 채점하는 구조는
구판 4.3절의 선별 전략 비교와 동일하다.

이 분석의 결론: 「무릎점은 효율 손실이 크다」는 전략의 성질이 아니라 k = 20 설정의
산물이다. 반면 가중합 − 효율 단독의 이득은 네 개의 k 모두에서 부호가 같아
(ΔY₂ +0.0858 ~ +0.1547) k 에 견고했다. 두 주장을 섞지 말 것.
"""
import numpy as np, json, pathlib, torch, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from phase2_model import MultiTaskCNN

ROOT = pathlib.Path(__file__).resolve().parent.parent
rng = np.random.default_rng(20260906)

D = dict(np.load(ROOT / "out/dataset.npz", allow_pickle=True))
P = np.load(ROOT / "out/phase3_points.npz", allow_pickle=True)
te = D["split"] == 2
y1, y2, y2v = D["y1"][te], D["y2"][te], D["y2v"][te]

model = MultiTaskCNN(D["X_bio"].shape[1])
model.load_state_dict(torch.load(ROOT / "out/model_lam050.pt", map_location="cpu"))
model.eval()
with torch.no_grad():
    a, b = model(torch.tensor(D["X_seq"][te]), torch.tensor(D["X_bio"][te]))
p1, p2 = a.numpy(), b.numpy()

norm = lambda v: (v - v.min()) / (v.max() - v.min())
n1, n2 = norm(p1), norm(p2)
F1 = P["f1"]
NF = len(F1)


def select(name, k):
    if name == "eff":     return np.argsort(-p1)[:k]
    if name == "fid":     return np.argsort(-p2)[:k]
    if name == "sum":     return np.argsort(-(0.5 * n1 + 0.5 * n2))[:k]
    if name == "knee":                      # 전선 위에서 정규화 합이 큰 순
        return F1[np.argsort(-(n1[F1] + n2[F1]))[:min(k, NF)]]
    raise ValueError(name)


LABEL = {"eff": "효율 단독", "fid": "유지율 단독",
         "knee": "파레토 무릎점", "sum": "가중합 (0.5Ŷ₁+0.5Ŷ₂)"}
KS = [5, 10, 20, 26]
R = {"n_front_F1": int(NF), "k_values": KS, "results": {}}

print(f"예측 파레토 전선 |F₁| = {NF}개\n")
hdr = f"{'k':>3} {'k/|F₁|':>7}  {'전략':22s} {'측정 Y₁':>16s} {'측정 Y₂':>16s} {'Y₂ᵥ(eSpCas9)':>14s}"
print(hdr); print("-" * len(hdr))
for k in KS:
    for m in ["eff", "fid", "knee", "sum"]:
        idx = select(m, k)
        rec = dict(k_effective=int(len(idx)),
                   Y1_mean=float(y1[idx].mean()), Y1_sd=float(y1[idx].std(ddof=1)),
                   Y2_mean=float(y2[idx].mean()), Y2_sd=float(y2[idx].std(ddof=1)),
                   Y2v_mean=float(y2v[idx].mean()))
        R["results"][f"k{k}_{m}"] = rec
        print(f"{k:>3} {k/NF:>7.2f}  {LABEL[m]:22s} "
              f"{rec['Y1_mean']:>8.4f} ± {rec['Y1_sd']:.3f} "
              f"{rec['Y2_mean']:>8.4f} ± {rec['Y2_sd']:.3f} "
              f"{rec['Y2v_mean']:>13.4f}")
    print()

# ---- 무릎점과 가중합의 차이가 k에 따라 어떻게 변하는가
print("무릎점 − 가중합 (측정값 차이)")
R["knee_minus_sum"] = {}
for k in KS:
    ik, isum = select("knee", k), select("sum", k)
    d1 = float(y1[ik].mean() - y1[isum].mean())
    d2 = float(y2[ik].mean() - y2[isum].mean())
    R["knee_minus_sum"][f"k{k}"] = dict(dY1=d1, dY2=d2)
    print(f"  k={k:>2} (전선의 {min(k,NF)/NF*100:>4.0f}%) : ΔY₁ {d1:+.4f}  ΔY₂ {d2:+.4f}")

# ---- 전선 위 가이드를 정규화 합 순으로 늘어놓았을 때의 측정 Y₁ 궤적
order = F1[np.argsort(-(n1[F1] + n2[F1]))]
R["front_ordered_measured"] = dict(
    rank=list(range(1, NF + 1)),
    meas_Y1=[float(x) for x in y1[order]],
    meas_Y2=[float(x) for x in y2[order]],
    pred_Y1=[float(x) for x in p1[order]],
    pred_Y2=[float(x) for x in p2[order]],
    cum_meas_Y1=[float(y1[order][:i + 1].mean()) for i in range(NF)],
    cum_meas_Y2=[float(y2[order][:i + 1].mean()) for i in range(NF)],
)
print("\n전선 상위 순위별 누적 평균 측정 Y₁ (k를 늘릴 때의 희석)")
for i in [0, 4, 9, 14, 19, 25]:
    print(f"  상위 {i+1:>2}개 누적: Y₁ {R['front_ordered_measured']['cum_meas_Y1'][i]:.4f}"
          f"  Y₂ {R['front_ordered_measured']['cum_meas_Y2'][i]:.4f}")

json.dump(R, open(ROOT / "out/phase7_knee_k.json", "w"), indent=2, ensure_ascii=False)
print("\n[save] out/phase7_knee_k.json")
