"""
논문 도판 생성 (light-mode 인쇄용 고정 팔레트)
=============================================
색은 검증된 기본 팔레트에서 역할별로만 가져온다.
  범주형 slot1 blue #2a78d6 / slot2 orange #eb6834 / slot3 aqua #1baf7a
  순차형   blue 100→700 ramp
  발산형   blue ↔ red, 중립 중점 gray #f0efec
"""
import numpy as np, json, pathlib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "fig"; FIG.mkdir(exist_ok=True)

for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"]:
    if pathlib.Path(p).exists():
        fm.fontManager.addfont(p)
plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False, "figure.dpi": 200, "savefig.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "svg.hashsalt": "ksef-crispr",   # SVG 내부 id 를 고정 → 재현 가능
})
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#8a8985"
LAMBDA_ADOPTED = 0.5      # 논문이 채택한 결합손실 가중치 (out/model_lam050.pt)
RED = "#e34948"   # 발산형 램프의 빨강 끝 — 그림 3에서 '예측 파레토 전선'에 배정된 색
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_br", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948", "#8f1f1e"])


VECTOR = True   # True 면 PNG 외에 PDF·SVG 도 함께 저장 (후편집용 벡터)


# 벡터판에 생성 시각이 박히면 내용이 같아도 md5 가 매번 달라져서
# "벡터가 낡았는지" 를 md5 로 판정할 수 없다. 날짜 메타데이터를 비워 재현 가능하게 만든다.
META = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", facecolor="#fcfcfb")
    if VECTOR:
        for ext in ("pdf", "svg"):
            fig.savefig(FIG / f"{name}.{ext}", facecolor="#fcfcfb", metadata=META[ext])
    plt.close(fig)
    print("  fig/" + name + ".png" + ("  (+pdf, +svg)" if VECTOR else ""))


# --------------------------------------------------- Fig 1  측정값 분포와 상관
def fig1(D):
    y1, y2 = D["y1"], D["y2"]
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.0), layout="constrained")
    for a, v, c, t in [(ax[0], y1, S1, "Y₁  WT-SpCas9 편집효율"),
                       (ax[1], y2, S2, "Y₂  HF1/WT 활성 유지율")]:
        a.hist(v, bins=60, color=c, edgecolor="none")
        a.set_title(t, color=INK); a.set_xlabel("측정값"); a.set_ylabel("가이드 수")
        a.grid(axis="y"); a.set_axisbelow(True)
        top = a.get_ylim()[1]
        a.set_ylim(0, top * 1.14)
        a.axvline(v.mean(), color=INK, lw=1.2, ls="--")
        a.annotate(f"평균 {v.mean():.3f}", (v.mean(), top * 1.08),
                   xytext=(5, 0), textcoords="offset points",
                   color=INK, fontsize=8, va="center")
    h = ax[2].hexbin(y1, y2, gridsize=45, cmap=SEQ, mincnt=1, linewidths=0)
    ax[2].set_xlabel("Y₁ 편집효율"); ax[2].set_ylabel("Y₂ 활성 유지율")
    r = np.corrcoef(y1, y2)[0, 1]
    ax[2].set_title(f"두 목표의 결합분포  (Pearson r = {r:+.3f})", color=INK)
    cb = fig.colorbar(h, ax=ax[2], pad=0.02); cb.set_label("가이드 수", color="#52514e")
    cb.outline.set_visible(False)
    save(fig, "fig1_distributions")



# ------------------------- 시드 스윕 집계 (그림 2·10 전용, 2026-09-13)
# λ 마다 시드 5개(0–4)의 평균과 표본표준편차(ddof=1). 집계는 src/phase2_seed_agg.py.
# out/phase2_lambda_sweep.json (표 1 = 원 실행) 은 읽지도 쓰지도 않는다.
def seed_agg():
    import sys as _s; _s.path.insert(0, str(ROOT / "src"))
    from phase2_seed_agg import aggregate, check
    rows = aggregate()
    bad = check(rows)
    if bad:
        raise SystemExit(f"[중단] 시드 집계가 검산표와 다르다: {bad}")
    lam = np.array([r["lam"] for r in rows], float)
    return (lam,
            np.array([r["eff_mean"] for r in rows], float),
            np.array([r["eff_sd"] for r in rows], float),
            np.array([r["fid_mean"] for r in rows], float),
            np.array([r["fid_sd"] for r in rows], float))


# ------------------------------------------------------- Fig 2  λ 스윕
def fig2(_unused=None):
    # 2026-09-13 — 점은 시드 5개 평균, 오차막대는 표본표준편차(ddof=1).
    lam, e, es, f, fs = seed_agg()
    lam = list(lam)
    fig, ax = plt.subplots(figsize=(5.6, 3.6), layout="constrained")
    ax.errorbar(lam, e, yerr=es, fmt="-o", color=S1, lw=2, ms=5,
                capsize=3, elinewidth=1.2, label="Head A — 효율 Y₁")
    ax.errorbar(lam, f, yerr=fs, fmt="-o", color=S2, lw=2, ms=5,
                capsize=3, elinewidth=1.2, label="Head B — 유지율 Y₂")
    # 2026-09-13 — 최댓값 라벨은 제거했다. λ 0.1–0.9 의 표본표준편차가 0.0022–0.0323 이라
    # 최댓값은 이웃 λ 와 구분되지 않는다. 숫자로 강조하면 원고 3.2절(「λ 사이의 차이는
    # 시드 잡음 수준」)과 그림이 충돌한다.
    # F5 — 논문이 채택한 λ 를 명시한다 (독자의 첫 질문이 "왜 0.5인가"이므로)
    j = lam.index(LAMBDA_ADOPTED)
    ax.axvline(LAMBDA_ADOPTED, color="#c9c8c2", lw=1.1, ls="--", zorder=1)
    ax.scatter([LAMBDA_ADOPTED, LAMBDA_ADOPTED], [e[j], f[j]], s=90, facecolor="none",
               edgecolor=INK, lw=1.3, zorder=7)
    # 채택 λ 라벨은 유지한다 — 성능 주장이 아니라 「어느 점을 대표 모델로 썼는가」의 표시다.
    # 값은 계열색으로 나눠 붙인다.
    ax.annotate(f"채택 λ = {LAMBDA_ADOPTED}",
                (LAMBDA_ADOPTED, min(e[j], f[j])), textcoords="offset points",
                xytext=(0, -30), ha="center", fontsize=7.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.28", fc="#fcfcfb", ec="#c9c8c2", lw=0.6))
    for y, c, dy in ((e[j], S1, 13), (f[j], S2, -20)):
        ax.annotate(f"{y:.4f}", (LAMBDA_ADOPTED, y), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=7.5, color=c)
    ax.set_xlabel("λ   —   결합손실  L = λ·L(Y₁) + (1−λ)·L(Y₂)")
    ax.set_ylabel("hold-out Spearman ρ")
    ax.set_title("결합손실 가중치에 따른 두 과제의 학습 경쟁", color=INK)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    # 축 범위는 '평균'으로만 잡는다. λ=0 의 Y₁ 은 sd 0.370 이라 오차막대가 축 밖으로
    # 나가는데, 여기에 맞춰 축을 넓히면 학습 구간(λ 0.1–0.9)의 차이가 안 보인다.
    # 그래서 축은 그대로 두고 그 막대만 잘리게 하고, 잘렸음을 아래 주석으로 밝힌다.
    lo, hi = min(min(e), min(f)) - 0.16, max(max(e), max(f)) + 0.10
    ax.set_ylim(lo, hi)
    cut = [(lam[i], es[i]) for i in range(len(lam)) if e[i] - es[i] < lo]
    for x, sd in cut:
        ax.annotate("▼", (x, lo), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=S1)
        ax.annotate(f"λ = {x:.0f} 의 Y₁ 오차막대는 축 아래로 잘림 (sd {sd:.3f})",
                    (x, lo), xytext=(8, 12), textcoords="offset points",
                    ha="left", fontsize=7, color="#52514e")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)
    save(fig, "fig2_lambda_sweep")


# ------------------------------------------------------- Fig 3  파레토 전선
def fig3(P, summ):
    pred, f1 = P["pred"], P["f1"]
    fronts = []
    import sys; sys.path.insert(0, str(ROOT / "src"))
    try:
        from phase3_pareto import non_dominated_sort          # 원래 경로 (torch 필요)
        fronts = non_dominated_sort(pred, max_fronts=5)
    except ImportError:                                        # torch 없을 때 동등 대체
        r = non_dominated_ranks(pred)                          # 전선 크기 일치 검증 완료
        fronts = [np.flatnonzero(r == i + 1) for i in range(5)]

    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9), layout="constrained")
    ax[0].hexbin(pred[:, 0], pred[:, 1], gridsize=48, cmap=SEQ, mincnt=1, linewidths=0)
    o = np.argsort(pred[f1, 0])
    ax[0].plot(pred[f1][o, 0], pred[f1][o, 1], "-", color="#e34948", lw=2, zorder=5)
    ax[0].scatter(pred[f1, 0], pred[f1, 1], s=26, color="#e34948",
                  edgecolor="#fcfcfb", linewidth=1.2, zorder=6, label=f"파레토 전선 F₁ ({len(f1)}개)")
    ax[0].set_xlabel("예측 Ŷ₁  편집효율"); ax[0].set_ylabel("예측 Ŷ₂  활성 유지율")
    ax[0].set_title("모델 예측공간의 파레토 전선 (hold-out)", color=INK)
    ax[0].legend(loc="lower left")

    # F4 — F₁ 은 좌 패널과 같은 강조색, F₂–F₅ 는 램프 전 구간에 균등 배치해 구분한다
    ramp = [SEQ(x) for x in np.linspace(0.88, 0.20, len(fronts) - 1)]
    cols = [RED] + ramp
    for i, (fr, c) in enumerate(zip(fronts, cols)):
        ax[1].scatter(pred[fr, 0], pred[fr, 1], s=11 if i == 0 else 8, color=c,
                      linewidths=0, label=f"F{i+1}", zorder=5 - i * 0.1)
    ax[1].set_xlabel("예측 Ŷ₁  편집효율"); ax[1].set_ylabel("예측 Ŷ₂  활성 유지율")
    ax[1].set_title("비지배 정렬 계층 F₁–F₅ (축은 좌 패널과 동일)", color=INK)
    ax[1].legend(loc="lower left", ncol=5, columnspacing=0.8, handletextpad=0.2)
    for a in ax:
        a.grid(); a.set_axisbelow(True)
    # F3 — 같은 양(예측 Ŷ₁·Ŷ₂)이므로 두 패널의 축 범위를 맞춘다
    ax[1].set_xlim(ax[0].get_xlim()); ax[1].set_ylim(ax[0].get_ylim())
    save(fig, "fig3_pareto")


# ------------------------------------------- Fig 4  위치별 saliency + 서열 문법
def fig4(M, s4):
    pos = np.arange(1, 21)
    ie = np.array(s4["pos_importance"]["eff"]); ifid = np.array(s4["pos_importance"]["fid"])
    ie, ifid = ie / ie.sum(), ifid / ifid.sum()
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.3), layout="constrained",
                           gridspec_kw={"width_ratios": [1.15, 1]})
    w = 0.4
    ax[0].bar(pos - w / 2, ie, w, color=S1, label="Head A — 효율 Y₁")
    ax[0].bar(pos + w / 2, ifid, w, color=S2, label="Head B — 유지율 Y₂")
    ax[0].set_xticks(pos); ax[0].set_xlabel("gRNA 위치 (5′ → 3′, PAM 인접 = 20)")
    ax[0].set_ylabel("정규화 saliency 기여도")
    ax[0].set_title("위치별 결정 기여도 (|∂Ŷ/∂x · x|)", color=INK)
    ax[0].grid(axis="y"); ax[0].set_axisbelow(True); ax[0].legend(ncol=2)

    nt = np.array(s4["nt_importance"]["fid"])                 # (4,20)
    v = np.abs(nt).max()
    im = ax[1].imshow(nt, cmap=DIV, vmin=-v, vmax=v, aspect="auto")
    ax[1].set_yticks(range(4), list("ACGT")); ax[1].set_xticks(range(1, 20, 2), range(2, 21, 2))
    ax[1].set_xlabel("gRNA 위치"); ax[1].set_ylabel("염기")
    ax[1].set_title("Y₂(유지율)에 대한 염기별 부호 있는 기여도", color=INK)
    cb = fig.colorbar(im, ax=ax[1], pad=0.02); cb.outline.set_visible(False)
    cb.set_label("saliency", color="#52514e")
    save(fig, "fig4_saliency")


# --------------------------------------------- Fig 5  in-silico mutagenesis
def fig5(s4):
    ism = np.array(s4["ism_mean"])                            # (20,4,2)
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.3), layout="constrained")
    titles = ["ΔŶ₁  편집효율 변화", "ΔŶ₂  활성 유지율 변화"]
    LW = 1.8
    BRACKET_PT = 7.0   # 모서리 ㄱ자 팔 길이 (pt) — 변 길이와 무관한 물리 길이
    marks = []
    for k in range(2):
        m = ism[:, :, k].T
        v = np.abs(m).max()
        im = ax[k].imshow(m, cmap=DIV, vmin=-v, vmax=v, aspect="auto")
        ax[k].set_yticks(range(4), list("ACGT"))
        ax[k].set_xticks(range(1, 20, 2), range(2, 21, 2))
        ax[k].set_xlabel("치환 위치"); ax[k].set_ylabel("치환 염기")
        cb = fig.colorbar(im, ax=ax[k], pad=0.02); cb.outline.set_visible(False)
        # F1 — 가장 강한 칸은 흰 테두리로만 표시하고, 값은 히트맵 밖(제목 줄)에 쓴다.
        #      예전 판은 라벨 상자가 인접 3–4칸을 덮었다.
        i, j = np.unravel_index(np.argmin(m), m.shape)
        marks.append((k, m, i, j))
        ax[k].set_title(f"{titles[k]}\n최소 칸 (흰 테두리) 위치 {j+1} → {'ACGT'[i]}   {m[i, j]:+.3f}",
                        color=INK, fontsize=9.5)
    fig.suptitle("가상 돌연변이 시뮬레이션 — 위치×염기 평균 효과",
                 color=INK, fontsize=10)

    # 테두리를 칸 '바깥'에 그린다.
    # matplotlib 은 선을 경로 중앙에 그리므로 lw 의 절반이 칸 안쪽을 덮는다.
    # 여기 히트맵은 aspect="auto" 로 20×4 를 그려 칸이 세로로 길다 — 가로 픽셀 폭이
    # 작아서 같은 선 굵기가 가로에서 훨씬 크게 먹힌다. 그래서 축별로 오프셋을 따로
    # 계산해 사각형을 절반 굵기만큼 키운다. clip_on=False 가 없으면 히트맵 경계에
    # 붙은 칸(위치 20 등)의 바깥쪽 선이 잘린다.
    fig.canvas.draw()          # constrained layout 확정 후 한 번만
    for k, m, i, j in marks:
        bb = ax[k].get_window_extent()
        dx = (LW / 2) / 72 * fig.dpi / (bb.width / m.shape[1])    # 데이터 단위(가로)
        dy = (LW / 2) / 72 * fig.dpi / (bb.height / m.shape[0])   # 데이터 단위(세로)
        x0, x1 = j - .5 - dx, j + .5 + dx
        y0, y1 = i - .5 - dy, i + .5 + dy
        w, h = x1 - x0, y1 - y0
        # 네 모서리에 짧은 ㄱ자만 그린다. 변 전체를 두르면 선이 이웃 칸을 따라
        # 길게 덮는다(측정: 이웃 칸의 10.3%). 팔 길이는 데이터 비율이 아니라
        # 물리 길이(pt)로 잡는다 — 칸이 40×112 px 로 세로로 길어서, 같은 비율을 쓰면
        # 세로 팔만 길어지고 좌우 이웃이 크게 덮인다. 변의 40% 를 상한으로 둔다.
        arm_x = min(BRACKET_PT / 72 * fig.dpi / (bb.width / m.shape[1]), 0.40 * w)
        arm_y = min(BRACKET_PT / 72 * fig.dpi / (bb.height / m.shape[0]), 0.40 * h)
        # 2026-09-13 — 흰 선만 그리면 히트맵 경계에 붙은 칸(왼쪽 패널의 위치 20)에서는
        # 바깥쪽 ㄱ자가 도판 배경(#fcfcfb)과 같은 색이라 보이지 않았다.
        # 축선색으로 한 겹 깔고(케이싱) 그 위에 흰 선을 얹어, 진한 칸 위에서도
        # 밝은 배경 위에서도 네 모서리가 모두 읽히게 한다.
        for cx, sx in ((x0, +1), (x1, -1)):
            for cy, sy in ((y0, +1), (y1, -1)):
                px = [cx + sx * arm_x, cx, cx]
                py = [cy, cy, cy + sy * arm_y]
                ax[k].plot(px, py, color="#c9c8c2", lw=LW + 1.2, solid_joinstyle="miter",
                           solid_capstyle="butt", clip_on=False, zorder=5)
                ax[k].plot(px, py, color="#fcfcfb", lw=LW, solid_joinstyle="miter",
                           solid_capstyle="butt", clip_on=False, zorder=6)
    save(fig, "fig5_ism")


# ------------------------------- Fig 6  사전 등록 비교 (직접 학습 vs 두 모델 비율)
def fig6(_unused=None):
    """2026-09-13 — 막대/점은 시드 베이스 5개({0,10,20,30,40}) 평균,
    오차막대는 표본표준편차(ddof=1). out/phase6_baseline.json(원 실행)은 읽지 않는다."""
    import sys as _s; _s.path.insert(0, str(ROOT / "src"))
    from phase6_seed_agg import aggregate, check
    A = aggregate()
    bad = check(A)
    if bad:
        raise SystemExit(f"[중단] Phase 6 시드 집계가 검산값과 다르다: {bad}")

    meth = ["BASE", "ST", "MT"]
    label = {"BASE": "BASE — 두 모델 예측값의 비",
             "ST": "ST — 유지율 직접 학습 (단일과제)",
             "MT": "MT — 유지율 직접 학습 (멀티태스크)"}
    col = {"BASE": S1, "ST": S2, "MT": S3}   # 범주형 슬롯 순서대로 고정
    conds = [("full", "전체 학습셋\n(43,149)"), ("sub10", "10% 학습셋\n(4,314)")]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.6), layout="constrained",
                           gridspec_kw={"width_ratios": [1, 1.1]})

    x = np.arange(len(conds)); w = 0.26
    for i, m in enumerate(meth):
        mu = [A[c]["overall"][m]["mean"] for c, _ in conds]
        sd = [A[c]["overall"][m]["sd"] for c, _ in conds]
        ax[0].bar(x + (i - 1) * w, mu, w, yerr=sd, capsize=3,
                  error_kw=dict(elinewidth=1.1, ecolor=INK), color=col[m], label=label[m])
        for xx, u, d in zip(x + (i - 1) * w, mu, sd):
            ax[0].annotate(f"{u:.3f}\n±{d:.3f}", (xx, u + d), xytext=(0, 3),
                           textcoords="offset points", ha="center", fontsize=6.8, color=INK)
    ax[0].set_xticks(x, [t for _, t in conds])
    ax[0].set_ylabel("hold-out Spearman ρ  (측정 Y₂ 대비)")
    ax[0].set_ylim(0, 0.92)
    ax[0].set_title("유지율 추정 방식별 정확도", color=INK)
    ax[0].grid(axis="y"); ax[0].set_axisbelow(True)
    ax[0].legend(loc="upper center", bbox_to_anchor=(1.08, -0.16), ncol=1)

    # 우 — 사분위별 MT − BASE. 시드별 최대 사분위가 갈리므로 오차막대가 크게 나온다.
    # 작아 보이게 다듬지 않는다. P2 가 판정 불가라는 사실이 그림에 드러나야 한다.
    series = {}
    for c, lab, sty, cc in [("full", "전체 학습셋", "-o", S1),
                            ("sub10", "10% 학습셋", "--s", S2)]:
        q = A[c]["quartile_gap_MT_minus_BASE"]
        series[c] = (q["mean"], q["sd"], cc)
        ax[1].errorbar(range(4), q["mean"], yerr=q["sd"], fmt=sty, lw=2, ms=6,
                       color=cc, capsize=3, elinewidth=1.1, label=lab)
    gf, sf, cf = series["full"]; gs, ss, cs = series["sub10"]
    for j in range(4):
        for val, sd, cc in ((gf[j], sf[j], cf), (gs[j], ss[j], cs)):
            up = val >= max(gf[j], gs[j])
            ax[1].annotate(f"{val:+.3f}", (j, val + (sd if up else -sd)),
                           xytext=(0, 10 if up else -16), textcoords="offset points",
                           ha="center", fontsize=7.5, color=cc)
    ax[1].axhline(0, color="#c9c8c2", lw=1)
    ax[1].set_xticks(range(4), [f"Q{i+1}" for i in range(4)])
    ax[1].set_xlabel("측정 WT 효율 사분위 (Q1 = 저효율)")
    ax[1].set_ylabel("MT − BASE  Spearman 격차")
    ax[1].set_xlim(-0.38, 3.38)
    lo_ = min(min(np.array(gf) - np.array(sf)), min(np.array(gs) - np.array(ss)))
    hi_ = max(max(np.array(gf) + np.array(sf)), max(np.array(gs) + np.array(ss)))
    ax[1].set_ylim(min(-0.01, lo_ - 0.022), hi_ + 0.028)
    ax[1].set_title("저효율 구간에서 격차가 커지는가 (P2)", color=INK)
    ax[1].grid(axis="y"); ax[1].set_axisbelow(True); ax[1].legend(loc="upper right")
    save(fig, "fig6_baseline")
    return A


# ==================================================================
#  그림 7–10 — 명세서 §4 "아직 그리지 않은 것"
# ==================================================================
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------- 공용 유틸
def non_dominated_ranks(pts):
    """두 목적 모두 최대화. 모든 점에 비지배 계층(1부터) 부여."""
    rank = np.full(len(pts), -1, dtype=int)
    rem = np.arange(len(pts)); r = 1
    while len(rem):
        P = pts[rem]
        ge = (P[None, :, 0] >= P[:, None, 0]) & (P[None, :, 1] >= P[:, None, 1])
        gt = (P[None, :, 0] >  P[:, None, 0]) | (P[None, :, 1] >  P[:, None, 1])
        dom = (ge & gt).any(axis=1)
        rank[rem[~dom]] = r; r += 1
        rem = rem[dom]
    return rank


def load_points():
    """hold-out 5,394개. scatter_points.csv 를 우선 읽고, 없으면 out/phase3_points.npz."""
    csv = ROOT / "data/scatter_points.csv"
    if csv.exists():
        d = pd.read_csv(csv)
        return (d[["pred_Y1", "pred_Y2"]].to_numpy(), d[["meas_Y1", "meas_Y2"]].to_numpy(),
                d["meas_Y2v_eSpCas9"].to_numpy(), np.flatnonzero(d["on_pareto_front"].to_numpy()))
    P = np.load(ROOT / "out/phase3_points.npz", allow_pickle=True)
    v = json.load(open(ROOT / "out/phase5_validation.json"))
    raise SystemExit("scatter_points.csv 없음 — eSpCas9 열이 필요하다")


# ------------------------------------- Fig 7  파레토 전선의 실측 재현 (명세 §4-4)
def fig7(seed=20260830):
    pred, meas, _, f1 = load_points()
    rank = non_dominated_ranks(meas)                 # 측정 공간 비지배 계층
    r26 = rank[f1]
    mfronts = [np.flatnonzero(rank == i) for i in (1,)]
    mf1 = mfronts[0]

    rng = np.random.default_rng(seed)
    B = 10000                      # 순열(재표집) 횟수 — 주석의 p 하한 1/B 를 결정한다
    null = np.array([np.median(rank[rng.choice(len(rank), len(f1), replace=False)])
                     for _ in range(B)])
    lo, hi = np.percentile(null, [2.5, 97.5])
    n_hit = int((null <= np.median(r26)).sum())
    p_one = n_hit / B

    fig, ax = plt.subplots(1, 2, figsize=(10.9, 4.0), layout="constrained",
                           gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) 측정 공간에서의 위치
    h = ax[0].hexbin(meas[:, 0], meas[:, 1], gridsize=48, cmap=SEQ, mincnt=1, linewidths=0)
    o = np.argsort(meas[mf1, 0])
    ax[0].plot(meas[mf1][o, 0], meas[mf1][o, 1], "-", color=S3, lw=1.8, zorder=5)
    ax[0].scatter(meas[mf1, 0], meas[mf1, 1], s=30, color=S3, edgecolor="#fcfcfb",
                  linewidth=1.1, zorder=6, label=f"측정값 기준 전선 F₁ᵐ ({len(mf1)}개)")
    ax[0].scatter(meas[f1, 0], meas[f1, 1], s=30, color=RED, edgecolor="#fcfcfb",
                  linewidth=1.1, zorder=7, label=f"예측 전선 F₁ 26개의 실측 위치")
    ax[0].set_xlabel("측정 Y₁  편집효율"); ax[0].set_ylabel("측정 Y₂  활성 유지율")
    ax[0].set_title("예측 전선 가이드는 실측 공간에서 어디에 있나", color=INK)
    ax[0].grid(); ax[0].set_axisbelow(True)
    hnd, lbl = ax[0].get_legend_handles_labels()
    ax[0].legend(hnd[::-1], lbl[::-1], loc="lower left", fontsize=8)
    cb = fig.colorbar(h, ax=ax[0], pad=0.02); cb.set_label("가이드 수", color="#52514e")
    cb.outline.set_visible(False)

    # (b) 실측 비지배 계층의 누적분포 — 무작위 기준선 대비
    xs = np.arange(1, rank.max() + 1)
    cdf_all = np.array([(rank <= x).mean() for x in xs])
    cdf_26 = np.array([(r26 <= x).mean() for x in xs])
    ax[1].step(xs, cdf_26, where="post", color=RED, lw=2, label="예측 전선 F₁ 26개")
    ax[1].step(xs, cdf_all, where="post", color=S1, lw=2, label="hold-out 전체 5,394개")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("측정값 기준 비지배 계층 (1 = 실측 전선)")
    ax[1].set_ylabel("누적 비율")
    ax[1].set_ylim(0, 1.02)
    ax[1].set_title("실측 계층 분포 — 예측 전선 vs 전체", color=INK)
    ax[1].grid(); ax[1].set_axisbelow(True); ax[1].legend(loc="lower right", fontsize=8)
    # 2026-09-13 — 순열 검정에서 0회가 나왔을 때 보고할 수 있는 값은 B 가 정한다.
    # 그래서 p 를 1/B 로 적고 B 와 초과 횟수를 함께 밝힌다.
    _sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    _exp = str(int(round(np.log10(B)))).translate(_sup)
    p_txt = (f"p < 1×10⁻{_exp} (순열 {B:,}회에서 {n_hit}회)"
             if n_hit == 0 else f"p = {p_one:.4f} (순열 {B:,}회에서 {n_hit}회)")
    txt = (f"중앙 계층  예측 전선 {np.median(r26):.1f}  vs  전체 {np.median(rank):.0f}\n"
           f"무작위 26개 중앙 계층 95% 구간  {lo:.0f}–{hi:.0f}\n"
           f"{p_txt}\n"
           f"단, 실측 F₁ 진입은 26개 중 {int((r26 == 1).sum())}개")
    ax[1].annotate(txt, (0.03, 0.97), xycoords="axes fraction", va="top", ha="left",
                   fontsize=7.5, color=INK,
                   bbox=dict(boxstyle="round,pad=0.35", fc="#fcfcfb", ec="#c9c8c2", lw=0.6))
    save(fig, "fig7_pareto_measured")
    return dict(median_rank_front=float(np.median(r26)), median_rank_all=float(np.median(rank)),
                null_ci=[float(lo), float(hi)], p_one_sided=p_one,
                n_perm=B, n_hit=n_hit,
                n_in_measured_f1=int((r26 == 1).sum()), max_rank=int(rank.max()))


# ------------------------------ Fig 8  eSpCas9 교차 효소 전이 (명세 §4-1)
def fig8():
    pred, meas, y2v, _ = load_points()
    pairs = [(pred[:, 1], y2v, "예측 Ŷ₂  (HF1 유지율 예측)", "모델 예측 → eSpCas9 실측"),
             (meas[:, 1], y2v, "측정 Y₂  (HF1 유지율 실측)", "HF1 실측 → eSpCas9 실측 (참조 기준)")]
    rhos = [spearmanr(a, b).statistic for a, b in [(p[0], p[1]) for p in pairs]]

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9), layout="constrained")
    hb = []
    for k, (x, y, xlab, title) in enumerate(pairs):
        h = ax[k].hexbin(x, y, gridsize=42, cmap=SEQ, mincnt=1, linewidths=0)
        hb.append(h)
        q = np.quantile(x, np.linspace(0, 1, 11))
        cx, cy = [], []
        for i in range(10):
            m = (x >= q[i]) & (x <= q[i + 1])
            if m.sum() > 10:
                cx.append(np.median(x[m])); cy.append(np.median(y[m]))
        ax[k].plot(cx, cy, "--o", color=INK, lw=1.4, ms=3.5, zorder=6)
        ax[k].set_xlabel(xlab); ax[k].set_ylabel("측정 Y₂ᵥ  eSpCas9 유지율")
        ax[k].set_title(f"{title}\nSpearman ρ = {rhos[k]:.4f}", color=INK, fontsize=9.5)
        ax[k].grid(); ax[k].set_axisbelow(True)
    vmax = max(h.get_array().max() for h in hb)
    for h in hb:
        h.set_clim(1, vmax)
    cb = fig.colorbar(hb[1], ax=ax[1], pad=0.02)
    cb.set_label("가이드 수 (공통 척도)", color="#52514e")
    cb.outline.set_visible(False)
    ax[0].annotate(f"참조 기준 대비 {rhos[0] / rhos[1] * 100:.1f}%\n(점선 = 10분위 중앙값)",
                   (0.03, 0.97), xycoords="axes fraction", va="top", fontsize=7.5, color=INK,
                   bbox=dict(boxstyle="round,pad=0.3", fc="#fcfcfb", ec="#c9c8c2", lw=0.6))
    fig.suptitle("학습에 쓰지 않은 효소(eSpCas9)로의 전이", color=INK, fontsize=10)
    save(fig, "fig8_espcas9_transfer")
    return dict(rho_pred=float(rhos[0]), rho_ceiling=float(rhos[1]),
                ratio=float(rhos[0] / rhos[1]))


# ------------------------------------- Fig 9  선별 전략 비교 (명세 §4-3, 표 6)
def fig9(k=20):
    pred, meas, y2v, f1 = load_points()
    p1, p2 = pred[:, 0], pred[:, 1]
    y1, y2 = meas[:, 0], meas[:, 1]
    nrm = lambda v: (v - v.min()) / (v.max() - v.min())
    n1, n2 = nrm(p1), nrm(p2)
    knee = f1[np.argsort(-(n1[f1] + n2[f1]))[:k]]
    strat = [("효율 단독\n(기존 도구 방식)", np.argsort(-p1)[:k]),
             ("유지율 단독", np.argsort(-p2)[:k]),
             # L-1 — 「본 프레임워크」를 무릎점에서 떼어 가중합으로 옮긴다.
             #        §4.5 가 실제 선별 규칙으로 권장하는 것은 가중합이고,
             #        무릎점은 k 에 민감해 k=20 에서 측정 Y₁ 이 0.5348 까지 내려간다.
             ("파레토 무릎점\n(k = 20)", knee),
             ("가중합 (본 프레임워크 권장)\n0.5·Ŷ₁+0.5·Ŷ₂", np.argsort(-(0.5 * n1 + 0.5 * n2))[:k]),
             ("hold-out 전체\n(n=5,394)", np.arange(len(y1)))]
    metrics = [("측정 Y₁ 편집효율", y1, S1), ("측정 Y₂ 유지율 (HF1)", y2, S2),
               ("측정 Y₂ᵥ 유지율 (eSpCas9)", y2v, S3)]

    fig, ax = plt.subplots(figsize=(8.6, 4.2), layout="constrained")
    x = np.arange(len(strat)); w = 0.26
    for i, (lab, v, c) in enumerate(metrics):
        vals = [v[idx].mean() for _, idx in strat]
        b = ax.bar(x + (i - 1) * w, vals, w, color=c, label=lab)
        ax.bar_label(b, fmt="%.3f", fontsize=7, padding=2, color=INK)
    ax.set_xticks(x, [s for s, _ in strat], fontsize=8)
    ax.set_ylabel("측정값 평균")
    ax.set_ylim(0, 1.34)
    ax.axvline(3.5, color="#c9c8c2", lw=1, ls=":")
    ax.set_title("선별 전략별 실측 성능 — 상위 20개 가이드 vs 전체 평균", color=INK)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    save(fig, "fig9_selection_strategies")
    return {s: {m[0]: float(m[1][idx].mean()) for m in metrics} for s, idx in strat}


# ------------------------------------------- Fig 10  λ 손실가중 전선 (명세 §4-2)
def fig10():
    # 2026-09-13 — 점은 시드 5개 평균, 양방향 오차막대는 표본표준편차(ddof=1).
    # 「비지배 λ」 표기는 전부 제거했다. 일곱 번의 산출에서 일곱 개의 서로 다른
    # 집합이 나왔고 공통 원소가 없어, 이 자료로 결정할 수 있는 양이 아니다 (4.7절).
    lam, e, es, f, fs = seed_agg()

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0), layout="constrained")
    for k, (a_, sel, title) in enumerate([
            (ax[0], np.ones_like(lam, bool), "전체 λ 11개"),
            (ax[1], (lam >= 0.1) & (lam <= 0.9), "λ = 0.1–0.9 확대 (양 끝 퇴화점 제외)")]):
        i_ = np.flatnonzero(sel)
        a_.errorbar(e[i_], f[i_], xerr=es[i_], yerr=fs[i_], fmt="o", color=S1,
                    ms=6, capsize=2.5, elinewidth=1.1, markeredgecolor="#fcfcfb",
                    markeredgewidth=0.9, zorder=5)
        # 오차막대가 생기면서 점들이 서로 겹친다 — λ 0.3·0.4 와 0.5·0.7 은 평균이
        # 거의 같은 자리다(그 자체가 이 그림의 내용이다). 라벨은 빈 곳으로 빼고
        # 지시선으로 제 점을 가리키게 한다. 손으로 맞춘 오프셋이다.
        off = {0.1: (0, 12), 0.2: (0, 12), 0.3: (-30, 18), 0.4: (0, 26),
               0.5: (31, 27), 0.6: (-31, -17), 0.7: (33, -21), 0.8: (15, 11),
               0.9: (0, -16), 0.0: (0, 12), 1.0: (0, 12)}
        lead = {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
        show = i_ if k == 1 else np.flatnonzero(np.isin(lam, [0.0, 1.0]))
        for i in show:
            key = round(float(lam[i]), 1)
            dx, dy = off.get(key, (0, 9))
            arrow = (dict(arrowstyle="-", color="#c9c8c2", lw=0.7,
                          shrinkA=0, shrinkB=4)
                     if (k == 1 and key in lead) else None)
            a_.annotate(f"λ = {lam[i]:.1f}" if k == 0 else f"{lam[i]:.1f}",
                        (e[i], f[i]), xytext=(dx, dy),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        color=INK, arrowprops=arrow)
        if k == 0:
            m_ = (lam >= .1) & (lam <= .9)
            a_.annotate("λ = 0.1–0.9\n(오른쪽 패널에서 확대)",
                        (e[m_].mean(), f[m_].mean()),
                        xytext=(-58, -46), textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK,
                        arrowprops=dict(arrowstyle="-", color="#c9c8c2", lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.3", fc="#fcfcfb", ec="#c9c8c2", lw=0.6))
        a_.set_xlabel("hold-out Spearman ρ  —  Y₁ 편집효율")
        a_.set_ylabel("hold-out Spearman ρ  —  Y₂ 유지율")
        a_.set_title(title, color=INK)
        a_.grid(); a_.set_axisbelow(True)
    ax[0].margins(0.14, 0.14)
    ax[1].margins(0.22, 0.28)
    fig.suptitle("λ 스윕의 목적공간 — 시드 5개 평균 ± 표본표준편차", color=INK, fontsize=10)
    save(fig, "fig10_lambda_front")
    return dict(lam=[float(v) for v in lam],
                eff_mean=[float(v) for v in e], eff_sd=[float(v) for v in es],
                fid_mean=[float(v) for v in f], fid_sd=[float(v) for v in fs],
                n_seeds=5, ddof=1, note="비지배 λ 표기는 재현되지 않아 제거함")


# ------------------------------- Fig 12  q-스캔 (귀무분포 대비 z)
# 자료: out/phase3_qscan_null.json — 실제 스피어만을 맞춘 가우시안 copula 귀무분포.
# 2026-09-06 원고 담당 지시로 세로축을 「관측/Nq²」에서 z 로 교체했다.
# Nq² 는 두 목적이 완전 독립일 때의 기대값이라 실제 상관(+0.032 / +0.068)을 무시한다.
# 번호 11 은 fig11_knee_k_sensitivity 가 쓰고 있어 12 로 둔다. 최종 번호는 원고 담당이 정한다.
def fig12():
    src = ROOT / "data/phase3_qscan_null.json"
    if not src.exists():
        src = ROOT / "out/phase3_qscan_null.json"
    J = json.load(open(src, encoding="utf-8"))

    def col(name, key):
        return np.array([r[key] for r in J["series"][name]["rows"]], float)

    q = col("measured", "q")
    zm, zp = col("measured", "z"), col("predicted", "z")
    band = J["sign_change"]["measured"]["z_crosses_zero_between"]   # [0.12, 0.10]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    ax.axvspan(max(band), min(band), color="#f0efec", zorder=0)
    ax.axhline(0, color="#c9c8c2", lw=1.4, zorder=1)
    for zz in (2, -2):
        ax.axhline(zz, color="#e8e7e3", lw=1.0, ls=":", zorder=1)
    ax.plot(q, zm, "-o", color=S1, lw=2, ms=5, zorder=4, label="측정값 기준")
    ax.plot(q, zp, "-s", color=S2, lw=2, ms=5, zorder=3, label="예측값 기준")

    ax.set_xlim(0.315, 0.021)                       # 역방향 — 오른쪽이 극단
    lo, hi = min(zm.min(), zp.min()), max(zm.max(), zp.max())
    ax.set_ylim(lo - 1.6, hi + 1.8)
    ticks = [0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{v:.2f}" for v in ticks])
    ax.set_xlabel("각 목적함수의 상위 분위 q   (오른쪽으로 갈수록 좁아짐)")
    ax.set_ylabel("z   (귀무분포 대비 표준화 편차)")
    ax.set_title("두 목적을 동시에 만족하는 가이드 수의 귀무분포 대비 편차", color=INK)
    ax.grid(axis="y"); ax.set_axisbelow(True)

    ax.annotate("z = ±2", (0.024, 2.0), xytext=(0, 5), textcoords="offset points",
                ha="right", fontsize=7.5, color="#52514e")
    # 직접 라벨은 측정 곡선 세 지점만 (전 지점 라벨 금지)
    for target, dy in ((0.15, -17), (0.10, -17), (0.05, 14)):   # 0.05 는 예측 곡선을 피해 위로
        i = int(np.argmin(np.abs(q - target)))
        ax.annotate(f"z = {zm[i]:+.2f}", (q[i], zm[i]), xytext=(0, dy),
                    textcoords="offset points", ha="center", fontsize=7.5, color=S1)
    ax.annotate(f"측정 곡선이 z = 0 을\n지나는 구간  q = {max(band):.2f}–{min(band):.2f}",
                (sum(band) / 2, ax.get_ylim()[1]), xytext=(0, -8),
                textcoords="offset points", ha="center", va="top",
                fontsize=7.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.28", fc="#fcfcfb", ec="#c9c8c2", lw=0.6))
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, "fig12_qscan")
    return dict(
        q=[float(v) for v in q],
        z_measured=[float(v) for v in zm], z_predicted=[float(v) for v in zp],
        null_model=J["null_model"], reps=int(J["reps"]), seed=int(J["seed"]), n=int(J["n"]),
        spearman={k: J["series"][k]["spearman"] for k in ("measured", "predicted")},
        sign_change=J["sign_change"])


# ------------------------------- Fig 11  무릎점 전략의 k 민감도
# 자료: out/phase7_knee_k.json (src/phase7_knee_k.py 산출)
# 2026-09-06 데이터 담당이 배포본 figures.py 에서 병합 — 원본과 동일한 코드
def fig11(R):
    ks = R["k_values"]; NF = R["n_front_F1"]
    meth = ["eff", "sum", "knee"]                      # 슬롯 순서대로 고정
    lab = {"eff": "효율 단독", "sum": "가중합 (0.5Ŷ₁+0.5Ŷ₂)", "knee": "파레토 무릎점"}
    col = {"eff": S1, "sum": S2, "knee": S3}
    mk = {"eff": "o", "sum": "s", "knee": "^"}
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.5), layout="constrained",
                           gridspec_kw={"width_ratios": [1, 1, 1.05]})

    for j, (key, ttl, ylab) in enumerate(
            [("Y1_mean", "측정 Y₁ — 편집 효율", "측정 Y₁"),
             ("Y2_mean", "측정 Y₂ — 활성 유지율", "측정 Y₂")]):
        for m in meth:
            v = [R["results"][f"k{k}_{m}"][key] for k in ks]
            ax[j].plot(ks, v, "-" + mk[m], color=col[m], lw=2, ms=6, label=lab[m])
        ax[j].set_xticks(ks); ax[j].set_xlabel("선별 개수 k")
        ax[j].set_ylabel(ylab); ax[j].set_title(ttl, color=INK)
        ax[j].grid(axis="y"); ax[j].set_axisbelow(True)
    ax[0].set_ylim(0.4, 1.0)
    # k=20 (논문 4.3절이 쓴 값) 표시 — 색이 아니라 선+라벨로
    for j in (0, 1):
        ax[j].axvline(20, color="#c9c8c2", lw=1, ls=":", zorder=0)
    ax[0].annotate("4.3절이 쓴 k = 20 (전선의 77%)", (20, 0.985), xytext=(-6, 0),
                   textcoords="offset points", ha="right", va="top",
                   fontsize=7.5, color=MUTED)

    # 우: 전선 순위별 누적 평균 — 희석이 일어나는 지점을 보인다
    f = R["front_ordered_measured"]
    rk = f["rank"]
    ax[2].plot(rk, f["cum_meas_Y1"], "-o", color=S1, lw=2, ms=3.5, label="누적 평균 측정 Y₁")
    ax[2].plot(rk, f["cum_meas_Y2"], "-s", color=S2, lw=2, ms=3.5, label="누적 평균 측정 Y₂")
    ax[2].axvline(20, color="#c9c8c2", lw=1, ls=":", zorder=0)
    ax[2].set_xlabel(f"전선 F₁ 내 순위 (정규화 합 기준, |F₁| = {NF})")
    ax[2].set_ylabel("상위 k개의 누적 평균", labelpad=2)
    ax[2].set_title("k를 늘릴수록 저효율 꼬리가 섞인다", color=INK)
    ax[2].grid(axis="y"); ax[2].set_axisbelow(True)
    ax[2].set_xticks([1, 5, 10, 15, 20, 26])
    ax[0].legend(loc="lower left", fontsize=8)
    ax[1].legend(loc="upper left", fontsize=8)
    ax[2].legend(loc="lower left", fontsize=8)
    save(fig, "fig11_knee_k_sensitivity")


# ==================================================================
if __name__ == "__main__":
    print("[figures]")

    def step(name, fn, *a):
        try:
            fn(*a)
        except FileNotFoundError as e:
            print(f"  [건너뜀] {name}: 입력 파일 없음 — {e.filename}")
        except ImportError as e:
            print(f"  [건너뜀] {name}: 모듈 없음 — {e.name}")

    # --- 그림 1–6 : out/ 산출물이 있어야 그려진다
    try:
        D = dict(np.load(ROOT / "out/dataset.npz", allow_pickle=True))
        fig1(D)
    except FileNotFoundError as e:
        print(f"  [건너뜀] fig1: 입력 파일 없음 — {e.filename}")
    step("fig2", fig2)
    step("fig3", lambda: fig3(np.load(ROOT / "out/phase3_points.npz", allow_pickle=True),
                              json.load(open(ROOT / "out/phase3_summary.json"))))
    step("fig4", lambda: fig4(None, json.load(open(ROOT / "out/phase4_summary.json"))))
    step("fig5", lambda: fig5(json.load(open(ROOT / "out/phase4_summary.json"))))
    step("fig6", fig6)

    # --- 그림 7–10 : data/scatter_points.csv + data/figure_data.json 으로 그린다
    step("fig7", fig7)
    step("fig8", fig8)
    step("fig9", fig9)
    step("fig10", fig10)
    step("fig11", lambda: fig11(json.load(open(ROOT / "out/phase7_knee_k.json", encoding="utf-8"))))
    step("fig12", fig12)
