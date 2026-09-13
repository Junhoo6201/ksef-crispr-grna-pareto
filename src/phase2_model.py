"""
Phase 2 — 멀티태스크 1D CNN + 결합 손실함수 λ 스윕
==================================================
구조
    입력 (4 x 20) one-hot
      └ 공유 은닉층 (Conv1d x3, BatchNorm, ReLU) → 국소 모티프 특징벡터
          + 11개 열역학/GC 스칼라 (train 통계로 표준화) 결합
            ├ Head A : Y1  WT 편집효율          (회귀)
            └ Head B : Y2  HF1/WT 활성 유지율   (회귀)

손실  L = λ·L_Y1 + (1-λ)·L_Y2      (λ ∈ {0.0, 0.1, ..., 1.0})
      λ를 훑으며 두 목표의 학습 경쟁 구조를 실험적으로 관찰한다.
"""
import numpy as np, torch, torch.nn as nn, json, pathlib, time
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _env import device

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEV = device()           # 기본 cpu = 원고 정본 조건. KSEF_DEVICE=auto 로 가속 선택
torch.set_num_threads(2)


class MultiTaskCNN(nn.Module):
    def __init__(self, n_bio=11, ch=128, emb=64):
        super().__init__()
        self.trunk = nn.Sequential(                      # 공유 표현층
            nn.Conv1d(4, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.merge = nn.Sequential(
            nn.Linear(ch + n_bio, emb * 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(emb * 2, emb), nn.ReLU(),
        )
        head = lambda: nn.Sequential(nn.Linear(emb, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_eff = head()      # Head A — 효율
        self.head_fid = head()      # Head B — 충실도 유지율

    def forward(self, x_seq, x_bio):
        h = self.pool(self.trunk(x_seq)).squeeze(-1)
        z = self.merge(torch.cat([h, x_bio], 1))
        return self.head_eff(z).squeeze(-1), self.head_fid(z).squeeze(-1)


def batches(n, bs, shuffle, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def run(lam, D, epochs=60, bs=256, seed=0, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    T = {k: {p: torch.tensor(D[k][D["split"] == s]).to(DEV)
             for p, s in [("tr", 0), ("va", 1), ("te", 2)]}
         for k in ["X_seq", "X_bio", "y1", "y2"]}

    model = MultiTaskCNN(D["X_bio"].shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    mse = nn.MSELoss()

    best, best_state, patience = 1e9, None, 0
    n_tr = len(T["y1"]["tr"])
    for ep in range(epochs):
        model.train()
        for b in batches(n_tr, bs, True, rng):
            b = torch.as_tensor(b, device=DEV)
            p1, p2 = model(T["X_seq"]["tr"][b], T["X_bio"]["tr"][b])
            loss = lam * mse(p1, T["y1"]["tr"][b]) + (1 - lam) * mse(p2, T["y2"]["tr"][b])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            p1, p2 = model(T["X_seq"]["va"], T["X_bio"]["va"])
            vl = (lam * mse(p1, T["y1"]["va"]) + (1 - lam) * mse(p2, T["y2"]["va"])).item()
        if vl < best - 1e-5:
            best, patience = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 12:
                break
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        p1, p2 = model(T["X_seq"]["te"], T["X_bio"]["te"])
    p1, p2 = p1.cpu().numpy(), p2.cpu().numpy()
    t1, t2 = T["y1"]["te"].cpu().numpy(), T["y2"]["te"].cpu().numpy()
    res = dict(lam=lam,
               sp_eff=float(spearmanr(p1, t1)[0]), sp_fid=float(spearmanr(p2, t2)[0]),
               r_eff=float(np.corrcoef(p1, t1)[0, 1]), r_fid=float(np.corrcoef(p2, t2)[0, 1]),
               mse_eff=float(((p1 - t1) ** 2).mean()), mse_fid=float(((p2 - t2) ** 2).mean()),
               epochs_run=ep + 1)
    return model, res


if __name__ == "__main__":
    D = dict(np.load(ROOT / "out/dataset.npz", allow_pickle=True))
    results, t0 = [], time.time()
    for lam in [round(x, 1) for x in np.arange(0, 1.01, 0.1)]:
        model, r = run(lam, D)
        results.append(r)
        print(f"λ={lam:.1f} | Spearman  eff {r['sp_eff']:+.4f}  fid {r['sp_fid']:+.4f} "
              f"| MSE eff {r['mse_eff']:.4f} fid {r['mse_fid']:.4f} | ep {r['epochs_run']:2d} "
              f"| {time.time()-t0:5.0f}s", flush=True)
        if lam == 0.5:
            torch.save(model.state_dict(), ROOT / "out/model_lam050.pt")
    json.dump(results, open(ROOT / "out/phase2_lambda_sweep.json", "w"), indent=2)
    print("[save] out/phase2_lambda_sweep.json, out/model_lam050.pt")
