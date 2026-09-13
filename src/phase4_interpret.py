"""
Phase 4 — Saliency Map + In-silico Mutagenesis
==============================================
(1) 그래디언트 기반 saliency
    ∂Ŷ/∂x  (x = one-hot 입력 텐서)를 역전파로 구하고, 실제 존재하는
    염기 위치의 성분만 취해(gradient × input) 위치·염기별 기여도를 만든다.
    Head A(효율)와 Head B(충실도)를 따로 계산해 두 목표의 서열 문법을 분리한다.

(2) In-silico mutagenesis (ISM)
    각 가이드의 20개 위치 × 4염기를 강제 치환한 뒤 예측값 변화를 측정한다.
    ΔŶ = Ŷ(치환) − Ŷ(원본).  saliency가 1차 근사인 데 반해 ISM은 유한차분
    실측이므로, 두 결과가 일치할 때 인과 주장이 강해진다.

(3) 극단점 대조
    파레토 전선의 양 끝(고효율·고충실도 / 고효율·저충실도)에서 ISM 지형이
    어떻게 달라지는지 비교한다.
"""
import numpy as np, torch, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from phase2_model import MultiTaskCNN

ROOT = pathlib.Path(__file__).resolve().parent.parent
NTS = list("ACGT")


def load_model(D):
    m = MultiTaskCNN(D["X_bio"].shape[1])
    m.load_state_dict(torch.load(ROOT / "out/model_lam050.pt", map_location="cpu"))
    m.eval()
    return m


# ------------------------------------------------------------- (1) saliency
def saliency(model, xs, xb, head):
    """gradient × input. 반환 (N, 4, 20)."""
    x = torch.tensor(xs, requires_grad=True)
    b = torch.tensor(xb)
    p1, p2 = model(x, b)
    (p1 if head == "eff" else p2).sum().backward()
    return (x.grad * x).detach().numpy()


# --------------------------------------------------- (2) in-silico mutagenesis
def ism(model, xs, xb, batch=512):
    """
    반환 (N, 20, 4, 2) — [가이드, 위치, 치환염기, (ΔY1, ΔY2)]
    원본 염기 자리는 0.
    """
    with torch.no_grad():
        b0 = model(torch.tensor(xs), torch.tensor(xb))
    base = np.stack([b0[0].numpy(), b0[1].numpy()], 1)          # (N,2)
    N = len(xs)
    out = np.zeros((N, 20, 4, 2), np.float32)
    for pos in range(20):
        for nt in range(4):
            xm = xs.copy()
            xm[:, :, pos] = 0.0
            xm[:, nt, pos] = 1.0
            preds = []
            with torch.no_grad():
                for i in range(0, N, batch):
                    q1, q2 = model(torch.tensor(xm[i:i + batch]),
                                   torch.tensor(xb[i:i + batch]))
                    preds.append(np.stack([q1.numpy(), q2.numpy()], 1))
            out[:, pos, nt, :] = np.concatenate(preds) - base
    orig = xs.argmax(1)                                          # (N,20)
    out[np.arange(N)[:, None], np.arange(20)[None, :], orig, :] = 0.0
    return out, base


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    D = dict(np.load(ROOT / "out/dataset.npz", allow_pickle=True))
    P = np.load(ROOT / "out/phase3_points.npz", allow_pickle=True)
    te = D["split"] == 2
    xs, xb, seq = D["X_seq"][te], D["X_bio"][te], D["seq"][te]
    pred = P["pred"]
    model = load_model(D)

    # --- 전체 test set saliency (위치별 평균 기여도)
    sal = {h: saliency(model, xs, xb, h) for h in ("eff", "fid")}
    pos_imp = {h: np.abs(v).sum(1).mean(0).tolist() for h, v in sal.items()}
    nt_imp = {h: v.mean(0).tolist() for h, v in sal.items()}     # (4,20)

    # --- 극단 부분집합 정의 (파레토 전선 양 끝 + 대조군)
    hi_e = pred[:, 0] >= np.quantile(pred[:, 0], 0.90)
    groups = {
        "high_eff_high_fid": hi_e & (pred[:, 1] >= np.quantile(pred[:, 1], 0.90)),
        "high_eff_low_fid":  hi_e & (pred[:, 1] <= np.quantile(pred[:, 1], 0.10)),
        "low_eff":           pred[:, 0] <= np.quantile(pred[:, 0], 0.10),
    }
    print({k: int(v.sum()) for k, v in groups.items()})

    # --- ISM: 전체 test set (5,394 x 80 forward pass)
    ism_all, base = ism(model, xs, xb)
    print(f"[ISM] {ism_all.shape}  ΔY1 range {ism_all[...,0].min():+.3f}~{ism_all[...,0].max():+.3f}"
          f"  ΔY2 range {ism_all[...,1].min():+.3f}~{ism_all[...,1].max():+.3f}")

    # 위치×염기 평균 효과 — 실제로 치환이 일어난 가이드에 한정한 조건부 평균
    # (원본 염기와 같은 칸은 변화가 0이므로 단순 평균은 효과를 희석시킨다)
    nz = (ism_all != 0).sum(0)                                   # (20,4,2)
    ism_mean = np.divide(ism_all.sum(0), nz, out=np.zeros_like(nz, float),
                         where=nz > 0)                           # (20,4,2)

    # --- 가장 강한 충실도 저해 치환 top-N (인과 후보)
    cand = []
    for pos in range(20):
        for nt in range(4):
            d2 = ism_all[:, pos, nt, 1]
            m = d2 != 0
            if m.sum() < 50:
                continue
            cand.append(dict(pos=pos + 1, nt=NTS[nt], n=int(m.sum()),
                             dY2=float(d2[m].mean()), dY1=float(ism_all[m, pos, nt, 0].mean())))
    cand.sort(key=lambda c: c["dY2"])
    print("\n[충실도를 가장 크게 떨어뜨리는 치환 top5]")
    for c in cand[:5]:
        print(f"  pos{c['pos']:2d}→{c['nt']}  ΔY2 {c['dY2']:+.4f}  ΔY1 {c['dY1']:+.4f}  (n={c['n']})")
    print("[충실도를 가장 크게 올리는 치환 top5]")
    for c in cand[-5:][::-1]:
        print(f"  pos{c['pos']:2d}→{c['nt']}  ΔY2 {c['dY2']:+.4f}  ΔY1 {c['dY1']:+.4f}  (n={c['n']})")

    # --- 가상 교란 검증: 최악 치환을 제거하면 Y2가 회복되는가
    worst = cand[0]
    pos_i, nt_i = worst["pos"] - 1, NTS.index(worst["nt"])
    has = xs[:, nt_i, pos_i] == 1
    rescue = {}
    if has.sum() >= 30:
        for alt in range(4):
            if alt == nt_i:
                continue
            xm = xs[has].copy(); xm[:, :, pos_i] = 0; xm[:, alt, pos_i] = 1
            with torch.no_grad():
                q1, q2 = model(torch.tensor(xm), torch.tensor(xb[has]))
            rescue[NTS[alt]] = dict(dY2=float((q2.numpy() - base[has, 1]).mean()),
                                    dY1=float((q1.numpy() - base[has, 0]).mean()))
        print(f"\n[교란 검증] pos{worst['pos']}에 {worst['nt']}를 가진 가이드 {int(has.sum())}개를 치환:")
        for k, v in rescue.items():
            print(f"   →{k}: ΔY2 {v['dY2']:+.4f}  ΔY1 {v['dY1']:+.4f}")

    np.savez(ROOT / "out/phase4_maps.npz",
             sal_eff=sal["eff"], sal_fid=sal["fid"],
             ism_mean=ism_mean, ism_all_pos=ism_all.mean(2),
             groups=np.stack([groups[k] for k in groups]),
             group_names=np.array(list(groups.keys())))
    json.dump(dict(pos_importance=pos_imp, nt_importance=nt_imp,
                   ism_mean=ism_mean.tolist(), candidates=cand[:20] + cand[-20:],
                   rescue=rescue, worst=worst,
                   group_sizes={k: int(v.sum()) for k, v in groups.items()}),
              open(ROOT / "out/phase4_summary.json", "w"), indent=2)
    print("\n[save] out/phase4_maps.npz, out/phase4_summary.json")
