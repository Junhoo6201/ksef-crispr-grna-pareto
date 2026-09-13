"""
Phase 1 — 데이터 벡터화 및 누출 없는 분할
=========================================
입력 : DeepHF (Wang et al. 2019) WT / eSpCas9(1.1) / SpCas9-HF1 실측 편집효율
출력 : out/dataset.npz  (X_seq, X_bio, y1, y2, y2v, split)

설계 결정
- Y1 = WT-SpCas9 편집효율               (on-target 활성)
- Y2 = HF1/WT 활성 유지율 (clip 0~1.5)   (고충실도 효소에서의 활성 보존 = 안전성 축)
- Y2v = eSpCas9/WT 유지율                (외부 검증용, 학습에 미사용)
- 분할: 서열 유사도(Hamming <= 3) 연결성분 단위 → train/valid/test
        무작위 분할 시 근접 서열이 train/test에 흩어져 성능이 과대평가된다.
"""
import numpy as np, pandas as pd, pathlib, json, collections

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _env import data_dir

ROOT = pathlib.Path(__file__).resolve().parent.parent
UP = data_dir()          # 경로 해석은 src/_env.py 로 일원화 (KSEF_DATA_DIR 로 재지정 가능)
RNG = np.random.default_rng(20260830)

BIO = ['ddg_1', 'dg_open', 'dg_bind20', 'dg_bind7to20',
       'gc_gt10', 'gc_lt10', 'gc_count', 'tm_global', 'tm2', 'tm3', 'dg_stem']
NT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


# ---------------------------------------------------------------- 1. 적재
def load():
    eff = pd.read_csv(UP / "deephf_3enzyme_derived.csv", index_col=0)
    eff.index.name = "seq20"
    bio = pd.read_csv(UP / "deephf_wt_decoded.csv").set_index("seq20")
    df = eff.join(bio[BIO], how="inner")
    df["Y1"] = df["wt"]
    df["Y2"] = (df["hf"] / df["wt"]).clip(0, 1.5)
    df["Y2v"] = (df["esp"] / df["wt"]).clip(0, 1.5)
    return df.reset_index()


# ------------------------------------------------- 2. 직교화 (one-hot 인코딩)
def one_hot(seqs):
    """(N, 4, 20) — 채널 우선(PyTorch Conv1d 규약). 각 염기·위치가 독립 피처."""
    a = np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), 20)
    idx = np.full(256, 255, np.uint8)
    for k, v in NT.items():
        idx[ord(k)] = v
    code = idx[a]
    X = np.zeros((len(seqs), 4, 20), np.float32)
    n, p = np.meshgrid(np.arange(len(seqs)), np.arange(20), indexing="ij")
    X[n, code, p] = 1.0
    return X


# ----------------------------------- 3. 서열 유사도 기반 분할 (leakage 차단)
def hamming_components(seqs, max_dist=3, n_blocks=4):
    """
    비둘기집 원리: 20개 위치를 4블록(각 5bp)으로 나누면
    Hamming <= 3 인 두 서열은 반드시 최소 한 블록이 완전히 같다.
    → 블록 해시 버킷 내부만 비교하면 O(N^2) 전수비교를 피할 수 있다.
    """
    n = len(seqs)
    arr = np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(n, 20)
    bounds = np.linspace(0, 20, n_blocks + 1).astype(int)

    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    n_pairs = 0
    for bi in range(n_blocks):
        lo, hi = bounds[bi], bounds[bi + 1]
        key = arr[:, lo:hi].tobytes()
        w = hi - lo
        buckets = collections.defaultdict(list)
        for i in range(n):
            buckets[key[i * w:(i + 1) * w]].append(i)
        for members in buckets.values():
            if len(members) < 2 or len(members) > 4000:
                continue
            m = np.array(members)
            sub = arr[m]
            for j in range(len(m)):
                d = (sub[j + 1:] != sub[j]).sum(1)
                hit = np.where(d <= max_dist)[0]
                for h in hit:
                    union(m[j], m[j + 1 + h])
                n_pairs += len(hit)
    comp = np.array([find(i) for i in range(n)])
    _, comp = np.unique(comp, return_inverse=True)
    return comp, n_pairs


def split_by_component(comp, frac=(0.80, 0.10, 0.10)):
    """연결성분 전체를 한 덩어리로 배정 → train/test 사이 Hamming<=3 쌍이 0개."""
    ids = np.unique(comp)
    RNG.shuffle(ids)
    sizes = np.array([(comp == c).sum() for c in ids])
    cum = np.cumsum(sizes) / sizes.sum()
    lab = np.where(cum <= frac[0], 0, np.where(cum <= frac[0] + frac[1], 1, 2))
    m = dict(zip(ids, lab))
    return np.array([m[c] for c in comp], np.int8)


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    df = load()
    print(f"[load] {len(df):,} guides  (20-mer unique: {df.seq20.nunique():,})")

    X_seq = one_hot(df.seq20.tolist())
    bio = df[BIO].values.astype(np.float32)

    comp, n_pairs = hamming_components(df.seq20.tolist())
    split = split_by_component(comp)
    print(f"[split] Hamming<=3 인접쌍 {n_pairs:,}개 → 연결성분 {comp.max()+1:,}개")
    for k, name in enumerate(["train", "valid", "test"]):
        print(f"         {name:5s} {int((split==k).sum()):6,d}")

    # 정규화 통계는 train에서만 산출 (test 정보 누출 방지)
    tr = split == 0
    mu, sd = bio[tr].mean(0), bio[tr].std(0) + 1e-8
    X_bio = (bio - mu) / sd

    (ROOT / "out").mkdir(exist_ok=True)
    np.savez_compressed(
        ROOT / "out/dataset.npz",
        X_seq=X_seq, X_bio=X_bio.astype(np.float32),
        y1=df.Y1.values.astype(np.float32),
        y2=df.Y2.values.astype(np.float32),
        y2v=df.Y2v.values.astype(np.float32),
        split=split, comp=comp,
        seq=df.seq20.values.astype("U20"),
        bio_mu=mu, bio_sd=sd,
    )
    json.dump({"n": len(df), "n_pairs": int(n_pairs),
               "n_components": int(comp.max() + 1),
               "bio_cols": BIO,
               "split_counts": {n: int((split == k).sum())
                                for k, n in enumerate(["train", "valid", "test"])}},
              open(ROOT / "out/phase1_summary.json", "w"), indent=2)
    print(f"[save] out/dataset.npz  X_seq{X_seq.shape} X_bio{X_bio.shape}")
    print(f"       Y1 mean {df.Y1.mean():.3f}  Y2 mean {df.Y2.mean():.3f}  "
          f"corr(Y1,Y2) pearson {np.corrcoef(df.Y1, df.Y2)[0,1]:+.4f}")
