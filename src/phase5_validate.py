"""
Phase 5 — 검증
==============
(V1) 외부 효소 일반화
     학습에 전혀 쓰지 않은 eSpCas9(1.1) 유지율(Y2v = esp/wt)에 대해
     Y2(HF1 기준) 예측이 얼마나 맞는지 본다. 서로 다른 고충실도 변이체에서
     같은 서열 문법이 작동한다면 Y2는 효소 하나의 특이현상이 아니다.

(V2) ISM 인과 주장의 실측 대조
     모델이 "충실도를 떨어뜨린다"고 지목한 치환을, 예측이 아니라 실측값에서
     직접 검증한다. 해당 위치·염기를 가진 가이드와 갖지 않은 가이드의
     측정 Y2를 비교(Mann–Whitney U)한다. hold-out만 사용.

(V3) 파레토 전선의 실측 재현
     예측 전선 위 가이드들의 '측정' Y1·Y2가 전체 분포 대비 실제로 우수한가.
"""
import numpy as np, json, pathlib, torch, sys
from scipy.stats import spearmanr, mannwhitneyu
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from phase2_model import MultiTaskCNN

ROOT = pathlib.Path(__file__).resolve().parent.parent
NTS = list("ACGT")

D = dict(np.load(ROOT / "out/dataset.npz", allow_pickle=True))
te = D["split"] == 2
xs, xb = D["X_seq"][te], D["X_bio"][te]
y1, y2, y2v = D["y1"][te], D["y2"][te], D["y2v"][te]

model = MultiTaskCNN(D["X_bio"].shape[1])
model.load_state_dict(torch.load(ROOT / "out/model_lam050.pt", map_location="cpu"))
model.eval()
with torch.no_grad():
    p1, p2 = model(torch.tensor(xs), torch.tensor(xb))
p1, p2 = p1.numpy(), p2.numpy()

R = {}

# ---------------------------------------------------------------- V1
R["V1_cross_enzyme"] = dict(
    rho_pred2_vs_HF1=float(spearmanr(p2, y2)[0]),
    rho_pred2_vs_eSpCas9=float(spearmanr(p2, y2v)[0]),
    rho_HF1_vs_eSpCas9_measured=float(spearmanr(y2, y2v)[0]),
    rho_pred1_vs_WT=float(spearmanr(p1, y1)[0]),
)
print("[V1] Y2 예측 → HF1 실측 ρ={rho_pred2_vs_HF1:+.4f} | "
      "학습에 없던 eSpCas9 실측 ρ={rho_pred2_vs_eSpCas9:+.4f} "
      "(두 실측 유지율끼리는 ρ={rho_HF1_vs_eSpCas9_measured:+.4f})".format(**R["V1_cross_enzyme"]))

# ---------------------------------------------------------------- V2
s4 = json.load(open(ROOT / "out/phase4_summary.json"))
tests = []
for c in s4["candidates"][:5] + s4["candidates"][-5:]:
    pos, nt = c["pos"] - 1, NTS.index(c["nt"])
    has = xs[:, nt, pos] == 1
    if has.sum() < 30:
        continue
    u, pv = mannwhitneyu(y2[has], y2[~has])
    tests.append(dict(pos=c["pos"], nt=c["nt"], n_with=int(has.sum()),
                      ism_dY2=c["dY2"],
                      meas_Y2_with=float(y2[has].mean()),
                      meas_Y2_without=float(y2[~has].mean()),
                      meas_diff=float(y2[has].mean() - y2[~has].mean()),
                      p=float(pv)))
R["V2_ism_vs_measured"] = tests
agree = sum(np.sign(t["ism_dY2"]) == np.sign(t["meas_diff"]) for t in tests)
sig = sum(t["p"] < 0.05 for t in tests)
R["V2_agreement"] = dict(n=len(tests), sign_agree=int(agree), p_lt_05=int(sig))
print(f"\n[V2] ISM 상위/하위 10개 치환의 실측 대조 — 부호 일치 {agree}/{len(tests)}, "
      f"p<0.05 {sig}/{len(tests)}")
for t in tests:
    mark = "O" if np.sign(t["ism_dY2"]) == np.sign(t["meas_diff"]) else "X"
    print(f"  {mark} pos{t['pos']:2d}·{t['nt']}  ISM ΔŶ₂ {t['ism_dY2']:+.4f}  "
          f"실측 차 {t['meas_diff']:+.4f}  (n={t['n_with']:4d}, p={t['p']:.2e})")

# ---------------------------------------------------------------- V3
P = np.load(ROOT / "out/phase3_points.npz", allow_pickle=True)
f1 = P["f1"]
R["V3_front_measured"] = dict(
    n_front=int(len(f1)),
    front_meas_Y1=float(y1[f1].mean()), all_meas_Y1=float(y1.mean()),
    front_meas_Y2=float(y2[f1].mean()), all_meas_Y2=float(y2.mean()),
    front_meas_Y2v=float(y2v[f1].mean()), all_meas_Y2v=float(y2v.mean()),
    pct_front_Y1=float((y1 < y1[f1].mean()).mean() * 100),
    pct_front_Y2=float((y2 < y2[f1].mean()).mean() * 100),
)
v = R["V3_front_measured"]
print(f"\n[V3] 예측 파레토 전선 {v['n_front']}개의 '실측' 성능")
print(f"     Y1 {v['front_meas_Y1']:.3f} vs 전체 {v['all_meas_Y1']:.3f} (상위 {100-v['pct_front_Y1']:.1f}%)")
print(f"     Y2 {v['front_meas_Y2']:.3f} vs 전체 {v['all_meas_Y2']:.3f} (상위 {100-v['pct_front_Y2']:.1f}%)")
print(f"     Y2v(eSpCas9) {v['front_meas_Y2v']:.3f} vs 전체 {v['all_meas_Y2v']:.3f}")

json.dump(R, open(ROOT / "out/phase5_validation.json", "w"), indent=2)
print("\n[save] out/phase5_validation.json")

# ---------------------------------------------------------------- V4
# 선별 전략 비교 — 프레임워크의 실용적 이득을 실측으로 보인다.
def norm(v):
    return (v - v.min()) / (v.max() - v.min())


k = 20
n1, n2 = norm(p1), norm(p2)
front_set = P["f1"]
knee = front_set[np.argsort(-(n1[front_set] + n2[front_set]))[:k]]
strategies = {
    "효율 단독 상위20 (기존 도구 방식)": np.argsort(-p1)[:k],
    "유지율 단독 상위20": np.argsort(-p2)[:k],
    "파레토 무릎점 상위20 (본 프레임워크)": knee,
    "가중합 상위20 (0.5·Ŷ₁+0.5·Ŷ₂)": np.argsort(-(0.5 * n1 + 0.5 * n2))[:k],
}
V4 = {}
print("\n[V4] 선별 전략별 '측정값' 성능 (hold-out 상위 20개)")
print(f"{'전략':38s} {'측정 Y₁':>8s} {'측정 Y₂':>8s} {'eSpCas9 Y₂v':>12s}")
for name, idx in strategies.items():
    V4[name] = dict(meas_Y1=float(y1[idx].mean()), meas_Y2=float(y2[idx].mean()),
                    meas_Y2v=float(y2v[idx].mean()),
                    meas_hf_abs=float((y1[idx] * y2[idx]).mean()))
    v = V4[name]
    print(f"{name:38s} {v['meas_Y1']:8.3f} {v['meas_Y2']:8.3f} {v['meas_Y2v']:12.3f}")
print(f"{'(전체 hold-out 평균)':38s} {y1.mean():8.3f} {y2.mean():8.3f} {y2v.mean():12.3f}")
R["V4_selection_strategies"] = V4
json.dump(R, open(ROOT / "out/phase5_validation.json", "w"), indent=2)
