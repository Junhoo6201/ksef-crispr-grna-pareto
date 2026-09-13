"""
Phase 3 — 파레토 전선 수학적 도출
=================================
두 종류의 전선을 구분해서 계산한다.

(A) 가이드 공간 전선  — 학습된 모델이 hold-out 가이드에 대해 예측한
    (Ŷ1, Ŷ2) 쌍을 2차원 유클리드 공간에 사상하고 비지배 정렬(fast
    non-dominated sort, Deb et al. 2002)로 F1, F2, ... 계층을 구한다.
    NSGA-II의 진화탐색은 쓰지 않는다. 후보 가이드 집합이 유한하고 전수
    열거가 가능하므로 정렬만으로 정확해가 나오며, 이 경우 진화탐색은
    불필요한 근사일 뿐이다.

(B) 손실가중 전선   — λ를 훑으며 얻은 모델들의 (Y1 성능, Y2 성능) 쌍.
    "모델이 두 목표 사이에서 무엇을 포기하는가"를 보여준다.

측정값 기준 전선도 같이 계산해 예측 전선과 대조한다.
"""
import numpy as np, torch, json, pathlib, pandas as pd
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from phase2_model import MultiTaskCNN

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------- 비지배 정렬 (fast non-dominated sort)
def dominates(a, b):
    """최대화 기준. a가 b를 지배 = 모든 목표에서 >= 이고 하나 이상에서 >."""
    return np.all(a >= b) and np.any(a > b)


def pareto_front(P):
    """단일 전선(F1) 인덱스. 목표별 정렬 + 스위핑으로 O(n log n)."""
    order = np.lexsort((-P[:, 1], -P[:, 0]))
    keep, best = [], -np.inf
    for i in order:
        if P[i, 1] > best:
            keep.append(i)
            best = P[i, 1]
    return np.array(keep)


def non_dominated_sort(P, max_fronts=None):
    """전 계층 분해. 남은 점에 반복적으로 전선을 벗겨낸다 (peeling)."""
    remaining = np.arange(len(P))
    fronts = []
    while len(remaining):
        f = pareto_front(P[remaining])
        fronts.append(remaining[f])
        remaining = np.delete(remaining, f)
        if max_fronts and len(fronts) >= max_fronts:
            break
    return fronts


def hypervolume_2d(front_pts, ref):
    """2목표 하이퍼볼륨 (기준점 ref 대비). 전선 품질의 스칼라 요약."""
    p = front_pts[np.argsort(-front_pts[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in p:
        if y > prev_y:
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y
    return float(hv)


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    D = dict(np.load(ROOT / "out/dataset.npz", allow_pickle=True))
    te = D["split"] == 2

    model = MultiTaskCNN(D["X_bio"].shape[1])
    model.load_state_dict(torch.load(ROOT / "out/model_lam050.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        p1, p2 = model(torch.tensor(D["X_seq"][te]), torch.tensor(D["X_bio"][te]))
    pred = np.stack([p1.numpy(), p2.numpy()], 1)
    meas = np.stack([D["y1"][te], D["y2"][te]], 1)
    seq = D["seq"][te]

    out = {}
    for name, P in [("predicted", pred), ("measured", meas)]:
        fronts = non_dominated_sort(P, max_fronts=8)
        ref = P.min(0)
        f1 = fronts[0]
        out[name] = dict(
            n_points=int(len(P)),
            front_sizes=[int(len(f)) for f in fronts],
            f1_size=int(len(f1)),
            f1_x_range=[float(P[f1, 0].min()), float(P[f1, 0].max())],
            f1_y_range=[float(P[f1, 1].min()), float(P[f1, 1].max())],
            hypervolume=hypervolume_2d(P[f1], ref),
            pearson=float(np.corrcoef(P[:, 0], P[:, 1])[0, 1]),
            spearman=float(spearmanr(P[:, 0], P[:, 1])[0]),
        )
        print(f"[{name:9s}] N={len(P):,}  F1={len(f1):3d}  "
              f"front sizes {out[name]['front_sizes'][:6]}  "
              f"corr(Y1,Y2) r={out[name]['pearson']:+.3f} ρ={out[name]['spearman']:+.3f}")

    # ------- 동시 최적 가이드가 실제로 존재하는가 (트레이드오프 보편성 점검)
    q = lambda v, p: v >= np.quantile(v, p)
    joint = {}
    for p in (0.75, 0.90, 0.95):
        n_pred = int((q(pred[:, 0], p) & q(pred[:, 1], p)).sum())
        n_meas = int((q(meas[:, 0], p) & q(meas[:, 1], p)).sum())
        exp = len(P) * (1 - p) ** 2                       # 완전 독립일 때 기대값
        joint[f"top{round((1-p)*100)}pct"] = dict(predicted=n_pred, measured=n_meas,
                                                expected_if_independent=round(exp, 1))
        print(f"  상위 {round((1-p)*100):2d}% 동시 만족: 예측 {n_pred:4d} / 측정 {n_meas:4d} "
              f"(독립 가정 기대값 {exp:.0f})")
    out["joint_optimal"] = joint

    # ------- (B) 손실가중 전선
    sweep = json.load(open(ROOT / "out/phase2_lambda_sweep.json"))
    S = np.array([[s["sp_eff"], s["sp_fid"]] for s in sweep])
    fl = pareto_front(S)
    out["lambda_front"] = dict(
        lambdas=[s["lam"] for s in sweep],
        sp_eff=[s["sp_eff"] for s in sweep], sp_fid=[s["sp_fid"] for s in sweep],
        non_dominated_lambdas=[sweep[i]["lam"] for i in sorted(fl)],
    )
    print(f"[λ-front] 비지배 λ: {out['lambda_front']['non_dominated_lambdas']}")

    # 전선 위 가이드 목록 저장 (논문 표 / Phase 4 입력)
    f1 = non_dominated_sort(pred, max_fronts=1)[0]
    pd.DataFrame({"seq20": seq[f1],
                  "pred_Y1": pred[f1, 0], "pred_Y2": pred[f1, 1],
                  "meas_Y1": meas[f1, 0], "meas_Y2": meas[f1, 1]}
                 ).sort_values("pred_Y1", ascending=False).to_csv(
        ROOT / "out/pareto_front_guides.csv", index=False)
    np.savez(ROOT / "out/phase3_points.npz", pred=pred, meas=meas, seq=seq,
             f1=f1, fronts=np.array([len(f) for f in non_dominated_sort(pred, 8)]))
    json.dump(out, open(ROOT / "out/phase3_summary.json", "w"), indent=2)
    print("[save] out/phase3_summary.json, out/pareto_front_guides.csv, out/phase3_points.npz")


# ---------------------------------------------------------------------------
# 부록 — 전선 크기의 기준값 (진단 대응, 2026-08-30 추가)
# ---------------------------------------------------------------------------
# 파레토 전선 크기는 두 축을 각각 단조변환해도 변하지 않는 순위(copula) 성질이다.
# 따라서 "예측 전선이 측정 전선보다 큰" 현상은 축 스케일이나 상한 절단으로 설명되지
# 않는다. 같은 스피어만 상관을 갖는 가우시안 copula에서 전선 크기를 반복 추정해
# 기준값을 만들고, 두 전선을 그 기준과 대조한다.
def front_size_reference(rho_s, n, n_rep=30, seed=0):
    """스피어만 rho_s, 표본 n 인 가우시안 copula의 전선 크기 분포."""
    rng = np.random.default_rng(seed)
    r = 2 * np.sin(np.pi * rho_s / 6)          # 스피어만 -> 피어슨 (가우시안)
    sizes = [len(pareto_front(rng.multivariate_normal([0, 0], [[1, r], [r, 1]], n)))
             for _ in range(n_rep)]
    return dict(mean=float(np.mean(sizes)), sd=float(np.std(sizes)),
                min=int(min(sizes)), max=int(max(sizes)), n_rep=n_rep)


def front_size_audit(pred, meas, out_path=None):
    from scipy.stats import rankdata
    rk = lambda A: np.column_stack([rankdata(A[:, 0]), rankdata(A[:, 1])])
    R = {}
    for name, A in [("predicted", pred), ("measured", meas)]:
        rho = float(spearmanr(A[:, 0], A[:, 1])[0])
        R[name] = dict(
            front_size=int(len(pareto_front(A))),
            front_size_rank_transformed=int(len(pareto_front(rk(A)))),
            spearman=rho,
            reference=front_size_reference(rho, len(A)),
        )
    R["independent_expectation_H_n"] = float(np.log(len(pred)) + 0.5772156649)
    if out_path:
        json.dump(R, open(out_path, "w"), indent=2)
    return R


if __name__ == "__main__" and "--audit" in __import__("sys").argv:
    D = dict(np.load(ROOT / "out/dataset.npz", allow_pickle=True))
    P = np.load(ROOT / "out/phase3_points.npz", allow_pickle=True)
    R = front_size_audit(P["pred"], P["meas"], ROOT / "out/phase3_front_audit.json")
    for k in ("predicted", "measured"):
        v = R[k]
        print(f"[{k:9s}] F1={v['front_size']:3d} (순위변환 후 {v['front_size_rank_transformed']:3d}) "
              f"| rho={v['spearman']:+.4f} | 기준값 {v['reference']['mean']:.1f}"
              f" ± {v['reference']['sd']:.1f} (범위 {v['reference']['min']}-{v['reference']['max']},"
              f" {v['reference']['n_rep']}회)")
    print(f"독립 가정 기대값 H_n = {R['independent_expectation_H_n']:.1f}")
