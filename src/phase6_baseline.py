"""
Phase 6 — 사전 등록된 비교: 직접 학습 vs 두 모델 비율
=====================================================
out/preregistration.md 의 P1·P2·P3 을 검증한다.

BASE : 단일과제 모델 2개(WT 효율, HF1 효율)를 따로 학습 → 예측값의 비
ST   : 단일과제 모델 1개가 유지율을 직접 학습
MT   : 멀티태스크 Head B (본 논문 λ=0.5 모델)

공정성: 세 방식 모두 동일 trunk·동일 하이퍼파라미터·동일 분할.
BASE 는 모델이 2개이므로 파라미터와 연산량이 2배 — 상쇄하지 않는다.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, json, pathlib, sys, time
from scipy.stats import spearmanr
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from phase2_model import MultiTaskCNN, batches
from _env import data_dir, device

ROOT = pathlib.Path(__file__).resolve().parent.parent
UP = data_dir()          # 하드코딩 절대경로 제거 (2026-09-12)
DEV = device()           # 기본 cpu = 원고 정본 조건
torch.set_num_threads(2)
CLIP = 1.5


class SingleTaskCNN(nn.Module):
    """MultiTaskCNN 과 완전히 동일한 trunk·merge, head 만 1개."""
    def __init__(self, n_bio=11, ch=128, emb=64):
        super().__init__()
        m = MultiTaskCNN(n_bio, ch, emb)
        self.trunk, self.pool, self.merge = m.trunk, m.pool, m.merge
        self.head = m.head_eff

    def forward(self, x_seq, x_bio):
        h = self.pool(self.trunk(x_seq)).squeeze(-1)
        return self.head(self.merge(torch.cat([h, x_bio], 1))).squeeze(-1)


def train_single(target, D, idx_tr, seed=0, epochs=60, bs=256, patience=12):
    """target: (N,) 학습 목표. idx_tr: 학습에 쓸 인덱스(부분표집 실험용)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    va = np.where(D["split"] == 1)[0]
    te = np.where(D["split"] == 2)[0]
    T = lambda k, i: torch.tensor(D[k][i]).to(DEV)
    y = lambda i: torch.tensor(target[i]).to(DEV)

    model = SingleTaskCNN(D["X_bio"].shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    mse = nn.MSELoss()
    xs_tr, xb_tr, y_tr = T("X_seq", idx_tr), T("X_bio", idx_tr), y(idx_tr)
    xs_va, xb_va, y_va = T("X_seq", va), T("X_bio", va), y(va)

    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for b in batches(len(idx_tr), bs, True, rng):
            b = torch.as_tensor(b, device=DEV)
            loss = mse(model(xs_tr[b], xb_tr[b]), y_tr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = mse(model(xs_va, xb_va), y_va).item()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_te = model(T("X_seq", te), T("X_bio", te)).cpu().numpy()
    return pred_te, ep + 1


def train_multi(D, idx_tr, lam=0.5, seed=0, epochs=60, bs=256, patience=12):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    va = np.where(D["split"] == 1)[0]
    te = np.where(D["split"] == 2)[0]
    T = lambda k, i: torch.tensor(D[k][i]).to(DEV)
    model = MultiTaskCNN(D["X_bio"].shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    mse = nn.MSELoss()
    xs, xb = T("X_seq", idx_tr), T("X_bio", idx_tr)
    a, b_ = T("y1", idx_tr), T("y2", idx_tr)
    xsv, xbv, av, bv = T("X_seq", va), T("X_bio", va), T("y1", va), T("y2", va)

    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for bi in batches(len(idx_tr), bs, True, rng):
            bi = torch.as_tensor(bi, device=DEV)
            p1, p2 = model(xs[bi], xb[bi])
            loss = lam * mse(p1, a[bi]) + (1 - lam) * mse(p2, b_[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            q1, q2 = model(xsv, xbv)
            vl = (lam * mse(q1, av) + (1 - lam) * mse(q2, bv)).item()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p1, p2 = model(T("X_seq", te), T("X_bio", te))
    return p1.cpu().numpy(), p2.cpu().numpy(), ep + 1


def score(pred, truth, name):
    return dict(method=name,
                spearman=float(spearmanr(pred, truth)[0]),
                pearson=float(np.corrcoef(pred, truth)[0, 1]),
                mse=float(((pred - truth) ** 2).mean()))


if __name__ == "__main__":
    D = dict(np.load(ROOT / "out/dataset.npz", allow_pickle=True))
    # 원자료에서 HF1 절대 효율을 그대로 가져온다 (y1*y2 는 절단 때문에 부정확)
    raw = pd.read_csv(UP / "deephf_3enzyme_derived.csv", index_col=0)
    hf = raw.loc[pd.Index(D["seq"]), "hf"].values.astype(np.float32)
    wt = D["y1"]
    te = np.where(D["split"] == 2)[0]
    y2_te = D["y2"][te]
    wt_te = wt[te]
    R = {"preregistered": True}
    t0 = time.time()

    for tag, frac in [("full", 1.0), ("sub10", 0.10)]:
        idx_all = np.where(D["split"] == 0)[0]
        if frac < 1.0:
            rs = np.random.default_rng(20260830)
            idx_tr = np.sort(rs.choice(idx_all, int(len(idx_all) * frac), replace=False))
        else:
            idx_tr = idx_all
        print(f"\n===== {tag}  (train N = {len(idx_tr):,}) =====", flush=True)

        p_wt, e1 = train_single(wt, D, idx_tr, seed=1)
        print(f"  [BASE-1] WT 단일과제 학습 완료  ep={e1}  {time.time()-t0:.0f}s", flush=True)
        p_hf, e2 = train_single(hf, D, idx_tr, seed=2)
        print(f"  [BASE-2] HF1 단일과제 학습 완료 ep={e2}  {time.time()-t0:.0f}s", flush=True)
        p_st, e3 = train_single(D["y2"], D, idx_tr, seed=3)
        print(f"  [ST]     유지율 직접 학습 완료   ep={e3}  {time.time()-t0:.0f}s", flush=True)
        p_mt1, p_mt2, e4 = train_multi(D, idx_tr, seed=4)
        print(f"  [MT]     멀티태스크 학습 완료    ep={e4}  {time.time()-t0:.0f}s", flush=True)

        base_ratio = np.clip(p_hf / np.maximum(p_wt, 1e-6), 0, CLIP)
        preds = {"BASE": base_ratio, "ST": p_st, "MT": p_mt2}

        rows = [score(v, y2_te, k) for k, v in preds.items()]
        # 참고: 두 모델의 개별 성능도 기록 (기준선이 약해서 진 것이 아님을 보이기 위해)
        aux = dict(base_wt=score(p_wt, wt_te, "BASE-WT"),
                   base_hf=score(p_hf, np.array(hf)[te], "BASE-HF1"),
                   mt_wt=score(p_mt1, wt_te, "MT-Y1"))

        # ---- P2: 측정 WT 효율 사분위별 격차
        q = np.quantile(wt_te, [0.25, 0.5, 0.75])
        strata = np.digitize(wt_te, q)
        strat = []
        for s in range(4):
            m = strata == s
            strat.append(dict(quartile=f"Q{s+1}", n=int(m.sum()),
                              wt_range=[float(wt_te[m].min()), float(wt_te[m].max())],
                              **{k: float(spearmanr(v[m], y2_te[m])[0]) for k, v in preds.items()}))
        R[tag] = dict(n_train=int(len(idx_tr)), overall=rows, aux=aux, strata=strat,
                      epochs=dict(base_wt=e1, base_hf=e2, st=e3, mt=e4))

        print(f"  {'method':6s} {'Spearman':>9s} {'Pearson':>9s} {'MSE':>8s}")
        for r in rows:
            print(f"  {r['method']:6s} {r['spearman']:9.4f} {r['pearson']:9.4f} {r['mse']:8.4f}")
        print(f"  (참고) BASE 개별: WT ρ={aux['base_wt']['spearman']:.4f}, "
              f"HF1 ρ={aux['base_hf']['spearman']:.4f} | MT Y1 ρ={aux['mt_wt']['spearman']:.4f}")
        print("  사분위별 Spearman (WT 효율 기준):")
        for s in strat:
            print(f"   {s['quartile']} WT[{s['wt_range'][0]:.3f}–{s['wt_range'][1]:.3f}] n={s['n']:4d} "
                  f"| BASE {s['BASE']:+.4f}  ST {s['ST']:+.4f}  MT {s['MT']:+.4f} "
                  f"| MT−BASE {s['MT']-s['BASE']:+.4f}")

    # ---------------- 사전 예측 판정
    f, s10 = R["full"], R["sub10"]
    g = lambda blk, m: next(r["spearman"] for r in blk["overall"] if r["method"] == m)
    gap_full = g(f, "MT") - g(f, "BASE")
    gap_sub = g(s10, "MT") - g(s10, "BASE")
    q_gaps = [s["MT"] - s["BASE"] for s in f["strata"]]
    verdict = dict(
        P1=dict(claim="MT > BASE (전체 Spearman)", gap=float(gap_full),
                supported=bool(gap_full > 0)),
        P2=dict(claim="WT 하위 25%(Q1)에서 격차 최대", quartile_gaps=[float(x) for x in q_gaps],
                argmax=f"Q{int(np.argmax(q_gaps))+1}", supported=bool(np.argmax(q_gaps) == 0)),
        P3=dict(claim="학습 10%에서 격차 확대", gap_full=float(gap_full), gap_sub10=float(gap_sub),
                supported=bool(gap_sub > gap_full)),
    )
    R["verdict"] = verdict
    print("\n===== 사전 예측 판정 =====")
    for k, v in verdict.items():
        print(f"  {k}: {'지지됨' if v['supported'] else '기각됨'}  — {v['claim']}")
    print(f"     P1 격차 {gap_full:+.4f} | P2 사분위 격차 {[round(x,4) for x in q_gaps]} "
          f"| P3 {gap_full:+.4f} → {gap_sub:+.4f}")

    json.dump(R, open(ROOT / "out/phase6_baseline.json", "w"), indent=2)
    print("\n[save] out/phase6_baseline.json")
