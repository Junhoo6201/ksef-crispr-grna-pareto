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
    # 수학 기호는 mathtext 로 쓴다 — 본문 LaTeX 수식과 같은 이탤릭 세리프가 되게
    "mathtext.fontset": "dejavuserif",
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


def save(fig, name, dpi=None):
    """dpi 를 주면 그 도판의 PNG 만 해당 해상도로 저장한다 (벡터는 무관).
    기본값은 rcParams 의 200 dpi — 다른 도판의 산출을 바꾸지 않기 위해서다."""
    fig.savefig(FIG / f"{name}.png", facecolor="#fcfcfb",
                **({"dpi": dpi} if dpi else {}))
    if VECTOR:
        for ext in ("pdf", "svg"):
            fig.savefig(FIG / f"{name}.{ext}", facecolor="#fcfcfb", metadata=META[ext])
    plt.close(fig)
    print("  fig/" + name + ".png" + ("  (+pdf, +svg)" if VECTOR else ""))


# --------------------------------------------------- Fig 1  측정값 분포와 상관
def fig1(D):
    y1, y2 = D["y1"], D["y2"]
    # 본문 그림 2 — 가로 170 mm 실치수 (기존 10.4 × 3.0 의 가로세로비 유지)
    fig, ax = plt.subplots(1, 3, figsize=(6.693, 6.693 * 3.0 / 10.4),
                           layout="constrained")
    for a, v, c, t in [(ax[0], y1, S1, r"$Y_1$  WT-SpCas9 편집효율"),
                       (ax[1], y2, S2, r"$Y_2$  HF1/WT 활성 유지율")]:
        a.hist(v, bins=60, color=c, edgecolor="none")
        a.set_title(t, color=INK, fontsize=9); a.set_xlabel("측정값", fontsize=8.5)
        a.set_ylabel("가이드 수", fontsize=8.5); a.tick_params(labelsize=8)
        a.grid(axis="y"); a.set_axisbelow(True)
        top = a.get_ylim()[1]
        a.set_ylim(0, top * 1.14)
        a.axvline(v.mean(), color=INK, lw=1.2, ls="--")
        a.annotate(f"평균 {v.mean():.3f}", (v.mean(), top * 1.08),
                   xytext=(5, 0), textcoords="offset points",
                   color=INK, fontsize=8, va="center")
    h = ax[2].hexbin(y1, y2, gridsize=45, cmap=SEQ, mincnt=1, linewidths=0)
    ax[2].set_xlabel(r"$Y_1$ 편집효율", fontsize=8.5)
    ax[2].set_ylabel(r"$Y_2$ 활성 유지율", fontsize=8.5); ax[2].tick_params(labelsize=8)
    r = np.corrcoef(y1, y2)[0, 1]
    # 170 mm 폭에서 원래 제목(「두 목표의 결합분포 (Pearson r = …)」)은 옆 패널 제목과 겹친다
    ax[2].set_title(f"결합분포  (Pearson r = {r:+.3f})", color=INK, fontsize=9)
    cb = fig.colorbar(h, ax=ax[2], pad=0.02)
    cb.set_label("가이드 수", color="#52514e", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    save(fig, "fig1_distributions", dpi=400)



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
    # 최댓값은 이웃 λ 와 구분되지 않는다. 숫자로 강조하면 원고 4.6절(「λ 사이의 차이는
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
    # 본문 그림 5 — 가로 170 mm 실치수 (기존 10.0 × 3.3 의 가로세로비 유지)
    fig, ax = plt.subplots(1, 2, figsize=(6.693, 6.693 * 3.3 / 10.0),
                           layout="constrained")
    titles = [r"$\Delta\hat{Y}_1$  편집효율 변화", r"$\Delta\hat{Y}_2$  활성 유지율 변화"]
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
    save(fig, "fig5_ism", dpi=400)


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
                   fontsize=8.0, color=INK,
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
    """선별 전략 비교(V4). **10쪽판 원고에 포함되지 않는다.**
    제외 사유는 분량이 아니다. 이 비교는 예측값으로 가이드를 고르므로
    모델을 다시 학습하면 결과가 무너진다. 가중합 − 효율 단독의 측정 Y₂ 이득은
    원 실행에서 +0.098(out/phase5_validation.json) 이었으나 사양을 기록한
    환경에서 전 과정을 재실행하자 +0.010 으로 약 1/10 이 되었다.
    같은 재실행에서 예측 전선은 26 → 20, 예측 기준 동시 만족 가이드는
    587·87·0 → 565·121·7 로 바뀌었다. 반면 측정값만으로 계산되는 양은
    한 자리도 움직이지 않았다. 코드는 기록으로 남긴다.
    """
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
    # 집합이 나왔고 공통 원소가 없어, 이 자료로 결정할 수 있는 양이 아니다 (4.6절).
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
                        fontsize=8.0, color=INK,
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
    """본문 그림 3 — 결합 상위 q 동시충족의 귀무분포 대비 z.
    (a) 전체 53,937 (측정값만 — 모델을 쓰지 않는 분석) / (b) 평가셋 5,394 (측정·예측).
    세로축 범위가 두 규모에서 크게 다르므로 한 축에 겹치지 않고 패널을 나눈다(이중 축 금지).
    가로 170 mm 실치수.
    """
    src = ROOT / "data/phase3_qscan_null.json"
    if not src.exists():
        src = ROOT / "out/phase3_qscan_null.json"
    J = json.load(open(src, encoding="utf-8"))
    srcF = ROOT / "data/qscan_fullsample.json"
    if not srcF.exists():
        srcF = ROOT / "out/qscan_fullsample.json"
    F = json.load(open(srcF, encoding="utf-8"))

    def col(name, key):
        return np.array([r[key] for r in J["series"][name]["rows"]], float)

    q = col("measured", "q")
    zm, zp = col("measured", "z"), col("predicted", "z")
    qf = np.array([r["q"] for r in F["rows"]], float)
    zf = np.array([r["z"] for r in F["rows"]], float)
    band = J["sign_change"]["measured"]["z_crosses_zero_between"]   # [0.12, 0.10]

    fig, ax = plt.subplots(1, 2, figsize=(6.693, 3.70), layout="constrained")
    TICKS = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]   # 13개 전부 쓰면 라벨이 겹친다

    def frame(a_, ylo, yhi):
        a_.axvspan(max(band), min(band), color="#f0efec", zorder=0)
        a_.axhline(0, color="#c9c8c2", lw=1.4, zorder=1)
        for zz in (2, -2):
            a_.axhline(zz, color="#e8e7e3", lw=1.0, ls=":", zorder=1)
        a_.set_xlim(0.315, 0.021)                   # 역방향 — 오른쪽이 극단
        a_.set_ylim(ylo, yhi)
        a_.set_xticks(TICKS); a_.set_xticklabels([f"{v:.2f}" for v in TICKS], fontsize=8)
        a_.set_xlabel("상위 분위 q   (오른쪽이 극단)", fontsize=8.5)
        a_.grid(axis="y"); a_.set_axisbelow(True)
        a_.tick_params(labelsize=8)

    # --- (a) 전체 표본
    frame(ax[0], -13.5, 20.5)
    ax[0].set_yticks([-10, -5, 0, 5, 10, 15, 20])
    ax[0].plot(qf, zf, "-o", color=S1, lw=1.8, ms=4, zorder=4, label="측정값 기준")
    ax[0].set_ylabel("z   (귀무분포 대비 표준화 편차)", fontsize=8.5)
    ax[0].set_title(f"(a) 전체 {F['n']:,}", fontsize=9, color=INK)
    for target, dx, dy, ha in ((0.30, 2, 8, "left"), (0.25, 5, 5, "left"),
                               (0.10, -5, 8, "right"), (0.05, 2, -15, "center")):
        i = int(np.argmin(np.abs(qf - target)))
        ax[0].annotate(f"{zf[i]:+.2f}", (qf[i], zf[i]), xytext=(dx, dy),
                       textcoords="offset points", ha=ha, fontsize=8.0, color=S1)

    # --- (b) 평가셋
    frame(ax[1], -6.5, 16.8)
    ax[1].set_yticks([-5, 0, 5, 10, 15])
    ax[1].plot(q, zm, "-o", color=S1, lw=1.8, ms=4, zorder=4, label="측정값 기준")
    ax[1].plot(q, zp, "-s", color=S2, lw=1.8, ms=4, zorder=3, label="예측값 기준")
    ax[1].set_title(f"(b) 평가셋 {J['n']:,}", fontsize=9, color=INK)
    for target, dx, dy, ha in ((0.15, -3, 7, "right"), (0.10, -4, -14, "right"),
                               (0.05, 1, 10, "center")):
        i = int(np.argmin(np.abs(q - target)))
        ax[1].annotate(f"{zm[i]:+.2f}", (q[i], zm[i]), xytext=(dx, dy),
                       textcoords="offset points", ha=ha, fontsize=8.0, color=S1)
    ax[1].legend(loc="lower left", fontsize=8.0)

    ax[0].annotate(f"z = 0 교차  q = {max(band):.2f}–{min(band):.2f}",
                   (sum(band) / 2, 20.5), xytext=(0, -6), textcoords="offset points",
                   ha="center", va="top", fontsize=8.0, color=INK)
    for a_ in ax:
        a_.annotate("z = ±2", (0.024, 2.0), xytext=(0, 3), textcoords="offset points",
                    ha="right", fontsize=8.0, color="#52514e")

    save(fig, "fig12_qscan", dpi=400)
    return dict(
        q=[float(v) for v in q],
        z_measured=[float(v) for v in zm], z_predicted=[float(v) for v in zp],
        q_full=[float(v) for v in qf], z_full=[float(v) for v in zf],
        null_model=J["null_model"], reps=int(J["reps"]), seed=int(J["seed"]), n=int(J["n"]),
        n_full=int(F["n"]),
        spearman={k: J["series"][k]["spearman"] for k in ("measured", "predicted")},
        spearman_full=float(F["spearman"]),
        sign_change=J["sign_change"])


# ------------------------- 본문 그림 4  비지배 전선 크기의 귀무분포와 관측값
# 자료: out/phase3_front_null.json + out/front_null_hist.json (평가셋)
#       out/front_null_fullsample.json (전체 표본)
# 주장은 「전선 크기가 우연 범위 안인가」이므로 산점도·파레토 곡선은 넣지 않는다.
# 관측은 1회이고 산포는 귀무분포의 것이므로 오차막대도 넣지 않는다.
def fig_front_null():
    def pick(name):
        p = ROOT / f"data/{name}"
        return p if p.exists() else ROOT / f"out/{name}"

    E = json.load(open(pick("phase3_front_null.json"), encoding="utf-8"))
    H = json.load(open(pick("front_null_hist.json"), encoding="utf-8"))
    FS = json.load(open(pick("front_null_fullsample.json"), encoding="utf-8"))

    fig, ax = plt.subplots(1, 2, figsize=(6.693, 3.25), layout="constrained")

    def shape(a_, xhi, title):
        a_.set_xlim(0, xhi); a_.set_xlabel("비지배 전선 크기 (가이드 수)", fontsize=8.5)
        a_.grid(axis="y"); a_.set_axisbelow(True); a_.tick_params(labelsize=8)
        a_.set_title(title, fontsize=9, color=INK)

    # ---------- (a) 전체 53,937 ----------
    c = np.array(FS["null_counts"], float)
    x = np.arange(len(c)); k = c > 0
    lo, hi = FS["tie_range"]
    ax[0].axvspan(lo, hi, color="#f0efec", zorder=0)
    ax[0].bar(x[k], c[k] / c.sum(), width=0.9, color="#cde2fb", edgecolor=S1,
              linewidth=0.6, zorder=2)
    ax[0].axvline(FS["independent_Hn"], color="#8a8985", lw=1.2, ls="--", zorder=3)
    ax[0].axvline(FS["observed_avgrank"], color=S1, lw=2.2, zorder=4)
    shape(ax[0], 35, f"(a) 전체 {FS['n']:,}")
    ax[0].set_ylabel("귀무 2,000회 중 비율", fontsize=8.5)
    ax[0].set_ylim(0, (c / c.sum()).max() * 1.34)          # 라벨 둘 자리
    top = ax[0].get_ylim()[1]
    BOX = dict(boxstyle="round,pad=0.28", fc="#fcfcfb", ec="#c9c8c2", lw=0.6)
    ax[0].annotate(f"귀무 {FS['null_mean']:.2f} ± {FS['null_sd']:.2f}  ({FS['reps']:,}회)\n"
                   "독립 $H_n$ = " + f"{FS['independent_Hn']:.2f}\n"
                   f"p = {FS['p_avgrank']:.4f}  (최불리 {FS['p_tie_min']:.4f})",
                   (0.98, 0.97), xycoords="axes fraction", ha="right", va="top",
                   fontsize=8.0, color=INK, linespacing=1.5, bbox=BOX, zorder=6)
    ax[0].annotate(f"관측 {FS['observed_avgrank']}", (FS["observed_avgrank"], top * 0.60),
                   xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                   fontsize=8.0, color=S1, fontweight="bold")
    ax[0].annotate(r"독립 $H_n$", (FS["independent_Hn"], top * 0.97),
                   xytext=(-3, 0), textcoords="offset points", ha="right", va="top",
                   fontsize=8.0, color="#52514e")
    ax[0].annotate(f"동점 무작위 분리 {lo}–{hi}", ((lo + hi) / 2, top * 0.42),
                   ha="center", va="center", fontsize=8.0, color="#52514e")

    # ---------- (b) 평가셋 5,394 ----------
    cm = np.array(H["null_counts"]["measured"], float)
    cp = np.array(H["null_counts"]["predicted"], float)
    cj = np.array(H["jitter_counts"]["1.00"], float)
    xx = np.arange(len(cm))
    ax[1].bar(xx, cm / cm.sum(), width=0.9, color="#cde2fb", edgecolor=S1, linewidth=0.6,
              zorder=2, label="귀무 (측정 조건)")
    ax[1].step(xx, cp / cp.sum(), where="mid", color=S2, lw=1.3, zorder=3,
               label="귀무 (예측 조건)")
    ax[1].step(xx, cj / cj.sum(), where="mid", color=MUTED, lw=1.3, ls=":", zorder=3,
               label="예측값 + 잔차 1.00× 잡음")
    shape(ax[1], 31, f"(b) 평가셋 {E['n']:,}")
    ax[1].set_ylim(0, max((cm / cm.sum()).max(), (cp / cp.sum()).max()) * 1.34)
    top1 = ax[1].get_ylim()[1]
    for nm, col_, lab, fy, ha_ in (("measured", S1, "측정", 0.72, "left"),
                                   ("predicted", S2, "예측", 0.97, "right")):
        v = E["null"][nm]["observed"]
        ax[1].axvline(v, color=col_, lw=2.2, zorder=4)
        ax[1].annotate(f"{lab} 관측 {v}   z = {E['null'][nm]['z']:+.2f}",
                       (v, top1 * fy), xytext=(4 if ha_ == "left" else -4, 0),
                       textcoords="offset points", ha=ha_, va="top",
                       fontsize=8.0, color=col_)
    ax[1].legend(loc="lower right", fontsize=8.0, frameon=True, facecolor="#fcfcfb",
                 edgecolor="#c9c8c2", framealpha=1.0, borderpad=0.4)

    save(fig, "fig_front_null", dpi=400)
    return dict(
        full=dict(n=FS["n"], observed=FS["observed_avgrank"], tie_range=FS["tie_range"],
                  null_mean=FS["null_mean"], null_sd=FS["null_sd"],
                  p=FS["p_avgrank"], p_worst=FS["p_tie_min"], Hn=FS["independent_Hn"]),
        eval=dict(n=E["n"], observed=E["observed"], null=E["null"],
                  jitter=E["jitter"], Hn=E["independent_theory_Hn"]))


# ------------------------------- Fig 11  무릎점 전략의 k 민감도
# 자료: out/phase7_knee_k.json (src/phase7_knee_k.py 산출)
# 2026-09-06 데이터 담당이 배포본 figures.py 에서 병합 — 원본과 동일한 코드
def fig11(R):
    """무릎점 k 민감도(phase7). **10쪽판 원고에 포함되지 않는다.**
    V4(fig9)와 함께 빠졌다. V4 가 빠진 사유는 fig9 의 주석을 볼 것.
    이 분석 자체의 결론은 따로 있다 — 구판 원고의 「무릎점은 효율 손실이
    크다」는 전략의 성질이 아니라 k = 20(|F₁| = 26 의 77%) 설정의 산물이다.
    무릎점 − 가중합의 측정 Y₂ 차는 k=5 −0.063 → k=26 +0.189 로 부호가 바뀐다.
    반대로 가중합 − 효율 단독의 이득은 k 에 견고했다(네 k 모두 ΔY₂ > 0,
    +0.0858 ~ +0.1547). 두 주장을 섞지 말 것.
    """
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
    # k=20 (구판 원고의 선별 전략 비교가 쓴 값. V4 와 함께 10쪽판에서 제외) 표시
    #   — 색이 아니라 선+라벨로
    for j in (0, 1):
        ax[j].axvline(20, color="#c9c8c2", lw=1, ls=":", zorder=0)
    ax[0].annotate("k = 20 (전선의 77%)", (20, 0.985), xytext=(-6, 0),
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
# ------------------------------------------- 파이프라인 개요 (번호 미부여)
# 연구 흐름 도식 — 가로 방향 플로차트. 본문 번호는 최종 조판에서 일괄 부여하므로
# 파일명에 번호를 넣지 않는다. 상자 안 수치는 전부 out/*.json 에서 코드로 읽는다.
BOX_FILL, BOX_EDGE = "#cde2fb", "#0b0b0b"      # 순차형 램프 1단계 / 검정 외곽선
BOX_FILL_END = "#9ec5f4"                       # 순차형 램프 2단계 — 종착 상자
BOX_FILL_SIDE = "#f0efec"                      # 발산형 중립 중점 — 보조(입출력) 상자
PLW = 2.2      # 상자 테두리 굵기
ALW = 2.0      # 화살표 굵기
AMS = 15       # 화살촉 크기


def _text_w(ax, t, fs, bold=False):
    """글자열의 렌더 폭을 축 좌표로 잰다 (상자 폭을 손으로 정하지 않기 위함)."""
    fig = ax.get_figure()
    h = ax.text(0.5, 0.5, t, fontsize=fs, fontweight="bold" if bold else None)
    fig.canvas.draw()
    bb = h.get_window_extent(); h.remove()
    inv = ax.transData.inverted()
    (x0, _), (x1, _) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return x1 - x0


def _box_w(ax, title, body="", ts=8.4, bs=7.2, padx=0.010):
    """상자 안 가장 긴 줄에서 필요한 상자 폭을 구한다."""
    w = _text_w(ax, title, ts, True)
    for ln in (body.split("\n") if body else []):
        w = max(w, _text_w(ax, ln, bs))
    return w + 2 * padx


def _row(ax, y, specs, x0, x1, min_gap=0.035):
    """한 줄에 상자들을 왼쪽부터 배치한다. 폭은 글자에서 자동으로 구하고,
    남는 자리를 같은 크기의 간격으로 나눈다. specs 는 _pbox 인자 dict 의 목록."""
    ws = [_box_w(ax, sp["title"], sp.get("body", ""),
                 sp.get("ts", 8.4), sp.get("bs", 7.2)) for sp in specs]
    gap = max(min_gap, (x1 - x0 - sum(ws)) / max(1, len(specs) - 1))
    out, x = [], x0
    for sp, w in zip(specs, ws):
        kw = {k: v for k, v in sp.items() if k not in ("title", "body")}
        out.append(_pbox(ax, x + w / 2, y, w, sp["title"], sp.get("body", ""), **kw))
        x += w + gap
    return out


def _pbox(ax, x, y, w, title, body="", h=None, fill=BOX_FILL, edge=BOX_EDGE,
          ts=8.4, bs=7.2):
    """중심 (x, y), 너비 w 의 둥근 상자. h 를 주지 않으면 글줄 수에서 자동으로 정한다
    — 위아래 여백 + 제목 + 빈 줄 + 본문 줄수. 상자 경계를 dict 로 반환."""
    from matplotlib.patches import FancyBboxPatch
    ax_h_pt = ax.get_figure().get_size_inches()[1] * 72.0      # 축 높이 1.0 = 몇 pt 인가
    pad = 8.0 / ax_h_pt
    t_h = ts * 1.35 / ax_h_pt
    blank = bs * 1.45 / ax_h_pt                                # 제목과 본문 사이 빈 줄
    line = bs * 1.60 / ax_h_pt
    n = len(body.split("\n")) if body else 0
    if h is None:
        h = pad * 2 + (t_h + blank + n * line if body else t_h)
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.008",
        linewidth=PLW, edgecolor=edge, facecolor=fill, zorder=2))
    if body:
        ax.text(x, y + h / 2 - pad, title, ha="center", va="top",
                fontsize=ts, fontweight="bold", color=INK, zorder=3)
        ax.text(x, y + h / 2 - pad - t_h - blank, body, ha="center", va="top",
                fontsize=bs, color=INK, linespacing=1.6, zorder=3)
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=ts, fontweight="bold", color=INK, zorder=3)
    return dict(x=x, y=y, w=w, h=h,
                L=x - w / 2, R=x + w / 2, B=y - h / 2, T=y + h / 2)


def _pline(ax, verts, lw=None, head=True):
    """꺾은선 경로를 따라가는 화살표. head=False 면 화살촉 없이 선만 (합류선용)."""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    path = Path(verts, [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1))
    ax.add_patch(FancyArrowPatch(
        path=path, arrowstyle="-|>" if head else "-", mutation_scale=AMS,
        linewidth=ALW if lw is None else lw,
        color=INK, shrinkA=0, shrinkB=0, joinstyle="miter", zorder=1))


def _pflow(ax, a, b, frac=0.5):
    """a 오른변 → b 왼변. 높이가 다르면 두 상자 사이에서 한 번 꺾는다."""
    if abs(a["y"] - b["y"]) < 1e-9:
        _pline(ax, [(a["R"], a["y"]), (b["L"], b["y"])])
    else:
        xm = a["R"] + (b["L"] - a["R"]) * frac
        _pline(ax, [(a["R"], a["y"]), (xm, a["y"]), (xm, b["y"]), (b["L"], b["y"])])


def _brace(ax, x_tip, x_arm, y0, y1, lw=2.4):
    """중괄호. x_tip 쪽으로 꼭지가 향하고 x_arm 쪽으로 두 팔이 벌어진다.
    두 개의 3차 베지에(위·아래 반쪽)로 그린다 — 제어점을 두 반쪽의 중간 x 에 두어
    세로 구간이 한 직선에 놓이게 한다."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    ym = (y0 + y1) / 2
    xc = x_tip + (x_arm - x_tip) * 0.5
    for ye in (y1, y0):
        path = Path([(x_arm, ye), (xc, ye), (xc, ym), (x_tip, ym)],
                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=INK,
                               linewidth=lw, capstyle="round", zorder=2))
    return (x_tip, ym)


def _brace_x(x_tip, x_arm, y0, y1, ys):
    """중괄호 곡선 위의 x 좌표를 y 로 조회한다 (갈래 화살표를 괄호에 붙이기 위함).
    _brace 와 같은 3차 베지에를 촘촘히 표본화해 보간한다."""
    ym = (y0 + y1) / 2
    xc = x_tip + (x_arm - x_tip) * 0.5
    t = np.linspace(0, 1, 512)[:, None]
    out = np.empty(len(ys), float)
    for ye, sel in ((y1, np.asarray(ys) >= ym), (y0, np.asarray(ys) < ym)):
        P = np.array([[x_arm, ye], [xc, ye], [xc, ym], [x_tip, ym]])
        B = ((1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1]
             + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3])
        bx, by = B[:, 0], B[:, 1]
        o = np.argsort(by)
        out[sel] = np.interp(np.asarray(ys)[sel], by[o], bx[o])
    return out


def _brace_fan(ax, heading, items, x_tip_l, x_arm_l, x_arm_r, x_tip_r, y_lo, y_hi,
               fs=7.2, hs=8.4, x_txt=None):
    """왼쪽 중괄호 → 항목별 화살표 → 오른쪽 중괄호. 예시 도식과 같은 형태.
    갈래 화살표는 괄호 곡선에서 바로 출발한다. 반환값은 (들어오는 점, 나가는 점)."""
    n = len(items)
    pad = (y_hi - y_lo) * 0.08
    ys = np.linspace(y_hi - pad, y_lo + pad, n)
    x_txt = x_arm_l + 0.030 if x_txt is None else x_txt
    ax.text((x_tip_l + x_tip_r) / 2, y_hi + 0.075, heading,
            ha="center", va="bottom", fontsize=hs, fontweight="bold", color=INK)
    p_in = _brace(ax, x_tip_l, x_arm_l, y_lo, y_hi)
    p_out = _brace(ax, x_tip_r, x_arm_r, y_lo, y_hi)
    xs = _brace_x(x_tip_l, x_arm_l, y_lo, y_hi, ys)
    for y, x0, t in zip(ys, xs, items):
        _pline(ax, [(x0, y), (x_txt - 0.006, y)])
        ax.text(x_txt, y, t, ha="left", va="center", fontsize=fs, color=INK)
    return p_in, p_out


def fig_pipeline_overview():
    """연구 흐름 도식 — 두 경로를 갈라놓은 2행 구조.

    윗줄(굵은 선)은 실측값만으로 성립하는 핵심 판정 경로, 아랫줄(가는 선)은
    모델을 거치는 보조 근거 경로다. 본문 폭 170 mm 실치수로 그리므로 조판해도
    글자 크기가 그대로다 (도판 1 pt = 지면 1 pt). 상자 폭은 글자에서 자동 계산한다.
    """
    P1 = json.load(open(ROOT / "out/phase1_summary.json", encoding="utf-8"))
    FD = json.load(open(ROOT / "out/figure_data.json", encoding="utf-8"))
    QS = json.load(open(ROOT / "out/phase3_qscan_null.json", encoding="utf-8"))
    SW = json.load(open(ROOT / "out/phase2_seed_sweep.json", encoding="utf-8"))
    S6 = json.load(open(ROOT / "out/phase6_seed_sweep.json", encoding="utf-8"))
    K7 = json.load(open(ROOT / "out/phase7_knee_k.json", encoding="utf-8"))

    n_all = P1["n"]; tr = P1["split_counts"]["train"]; te = P1["split_counts"]["test"]
    n_lam = len({r["lam"] for r in SW}); n_seed = len({r["seed"] for r in SW})
    n_base = len(S6["seed_bases"])
    f1 = FD["fig3_pareto"]["f1_size"]
    ks = "·".join(str(k) for k in K7["k_values"])
    M = QS["series"]["measured"]
    rho = M["spearman"]; reps = QS["reps"]; n_q = len(QS["q_values"])
    zof = {r["q"]: r["z"] for r in M["rows"]}
    qa, qb = QS["sign_change"]["measured"]["z_crosses_zero_between"]
    QHI, QLO = 0.25, 0.05                      # 판정 상자에 인용할 두 지점
    z_hi, z_lo = zof[QHI], zof[QLO]

    W_IN = 170 / 25.4                          # 본문 폭 실치수
    fig, ax = plt.subplots(figsize=(W_IN, 4.60), layout="constrained")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor("#fcfcfb")
    fig.canvas.draw()                          # 글자 폭 측정을 위해 먼저 그린다
    TS, BS = 9.0, 8.0      # 지면에서 제목 9 pt · 본문 8 pt
    BOLD, THIN = 3.0, 1.5                      # 핵심 경로 / 보조 경로 선 굵기
    YA, YB = 0.832, 0.470                      # 윗줄 / 아랫줄 중심 높이

    def note(x, y, t, ha="left", fs=7.4, color="#52514e", weight=None):
        ax.text(x, y, t, ha=ha, va="bottom", fontsize=fs, color=color,
                fontweight=weight, linespacing=1.5, zorder=3)

    # ================= 윗줄 — 실측값만으로 성립하는 핵심 판정 경로 =================
    a1, a2, a3, a4 = _row(ax, YA, [
        dict(title="Phase 1 — 데이터", ts=TS, bs=BS,
             body=f"Wang et al. 2019 보충자료\nn = {n_all:,} → 평가셋 {te:,}"),
        dict(title="실측 Y₁ · Y₂", ts=TS, bs=BS, body="모델 미사용"),
        dict(title=f"q-스캔 {n_q}점", ts=TS, bs=BS,
             body=f"결합 상위 q 동시충족\ncopula 귀무 {reps:,}회"),
        dict(title="★ 판정 근거", ts=TS, bs=BS, fill=BOX_FILL_END,
             body=f"전역 Spearman ρ = {rho:+.3f} (거의 무상관)\n"
                  f"z  {z_hi:+.2f} (q={QHI:.2f}) → {z_lo:+.2f} (q={QLO:.2f})\n"
                  f"부호 전환 q ≈ {qa:.2f} – {qb:.2f}"),
    ], x0=0.006, x1=0.994)

    for u, v in ((a1, a2), (a2, a3), (a3, a4)):
        _pline(ax, [(u["R"], YA), (v["L"], YA)], lw=BOLD)

    # ================= 아랫줄 — 모델을 거치는 보조 근거 경로 =================
    b1, = _row(ax, YB, [
        dict(title="Phase 2 — 멀티태스크 CNN", ts=TS, bs=BS,
             body=f"λ {n_lam} × 시드 {n_seed} = {n_lam * n_seed}개 모델\n"
                  f"채택 λ = {LAMBDA_ADOPTED}"),
    ], x0=0.006, x1=0.006)

    fan = [f"Phase 3   파레토 전선 F₁ = {f1}개",
           "Phase 4   saliency · ISM 20 위치 × 4 염기",
           f"Phase 7   무릎점 k 민감도 (k = {ks})",
           "Phase 5   외부·실측 검증 V1 · V2 · V4",
           "Phase 6   V5 사전등록 기준선 (BASE/ST/MT)"]
    x_txt = b1["R"] + 0.100
    x_end = x_txt + max(_text_w(ax, t, BS) for t in fan)
    ALW_SAVE = globals()["ALW"]; globals()["ALW"] = THIN
    f_in, f_out = _brace_fan(
        ax, "", fan, x_tip_l=b1["R"] + 0.040, x_arm_l=b1["R"] + 0.070,
        x_arm_r=x_end + 0.012, x_tip_r=x_end + 0.042,
        y_lo=0.268, y_hi=0.706, fs=BS, x_txt=x_txt)
    globals()["ALW"] = ALW_SAVE

    _pline(ax, [(b1["R"], YB), (f_in[0], YB)], lw=THIN)
    _pline(ax, [(f_out[0], YB), (a4["x"], YB), (a4["x"], a4["B"])], lw=THIN)
    note((f_out[0] + a4["x"]) / 2, YB + 0.016, "보조 근거", ha="center")

    # Phase 1 의 같은 데이터가 모델 경로로도 내려간다
    _pline(ax, [(a1["x"], a1["B"]), (b1["x"], b1["T"])], lw=THIN)
    note(a1["x"] + 0.010, (a1["B"] + b1["T"]) / 2 - 0.010, f"학습 {tr:,}", ha="left")

    # ================= 재현성 띠 =================
    st_b = (f"시드 반복 — λ 스윕 {n_lam} × 시드 {n_seed} = {n_lam * n_seed}개 모델 · "
            f"Phase 6 시드 베이스 {n_base}개\n"
            f"결정성 대조 — 같은 시드로 조건 A ×2 / 조건 B ×2")
    _pbox(ax, 0.500, 0.105, _box_w(ax, "재현성", st_b, ts=TS, bs=BS), "재현성", st_b,
          fill=BOX_FILL_SIDE, ts=TS, bs=BS)

    # ================= 범례 =================
    note(0.006, 0.968, "굵은 선 — 모델을 거치지 않는 경로 (핵심 판정)",
         fs=7.6, color=INK, weight="bold")
    note(0.470, 0.968, "가는 선 — 모델을 거치는 경로 (보조 근거)", fs=7.6)

    # 상자가 지면 밖으로 나가면 조용히 잘린 채 저장되므로 여기서 잡는다
    fig.canvas.draw()
    inv = ax.transData.inverted()
    over = []
    for pt in ax.patches:
        if type(pt).__name__ != "FancyBboxPatch":
            continue
        bb = pt.get_window_extent()
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        if x0 < -1e-6 or x1 > 1 + 1e-6 or y0 < -1e-6 or y1 > 1 + 1e-6:
            over.append((round(x0, 3), round(x1, 3), round(y0, 3), round(y1, 3)))
    if over:
        raise SystemExit(f"[중단] 상자가 그림 밖으로 나감: {over}")

    save(fig, "fig_pipeline_overview")


# ============================ 본문 그림 1 — 연구 설계 개요 ============================
# 명세: claude/도판-명세-그림1-파이프라인-2026-10-14.md
# 누적 교정: 도판-교정-그림1-2026-10-14.md · …-3차-2026-10-14.md
# 내용 갱신: 도판-지시-그림1코드화·기호통일-2026-10-28.md §2-2 (현행 원고 4장 기준)
#
# 명세의 규칙 셋을 그대로 지킨다 —
#   (1) 결과를 넣지 않는다 (상관계수·z·성능·전선 크기·결론 문장 금지)
#   (2) 상자 안에 그래프를 넣지 않는다
#   (3) 상자마다 제목 + 3줄 이내 (상자 3만 측정값 경로 설명을 위해 5줄)
# 측정값 경로는 굵은 선 + 진한 색, 예측값 경로는 가는 선 + 회색 — 색만으로 구분하지 않는다.
G1_W_MM, G1_H_MM = 170.0, 148.0          # 조판본이 잡아 둔 자리
G1_TS, G1_BS = 8.5, 8.0                  # 제목 / 본문 (지시서 하한 8 pt)
G1_LS = 1.45                             # 행간
G1_PAD, G1_TGAP = 4.0, 5.0               # 상자 안 위아래 여백 / 제목·본문 사이


def _g1_h(lines, note=False, ts=G1_TS, bs=G1_BS):
    """상자 높이를 pt 로 먼저 구한다 — 그림 높이를 내용에서 역산하기 위해서다."""
    h = G1_PAD * 2 + ts * 1.35
    if lines:
        # 실제 렌더 높이는 (줄수-1)*행간 + 글자높이 — len(lines)*행간 은 과대평가다
        h += G1_TGAP + ((len(lines) - 1) * G1_LS + 1.05) * bs
    if note:
        h += bs * G1_LS * 1.05
    return h


def _g1_box(ax, x, y_top, w, title, lines, fill, edge, lw, H, note=None,
            ts=G1_TS, bs=G1_BS):
    """왼쪽 위 모서리를 (x, y_top) 에 두고 상자를 그린다. y 는 축 좌표, H 는 축 높이(pt)."""
    from matplotlib.patches import FancyBboxPatch
    pad, tgap = G1_PAD / H, G1_TGAP / H
    th, lh = ts * 1.35 / H, bs * G1_LS / H
    h = _g1_h(lines, note is not None, ts, bs) / H
    ax.add_patch(FancyBboxPatch((x, y_top - h), w, h,
                                boxstyle="round,pad=0,rounding_size=0.006",
                                linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y_top - pad, title, ha="center", va="top",
            fontsize=ts, fontweight="bold", color=INK, zorder=3)
    if lines:
        ax.text(x + 0.012, y_top - pad - th - tgap, "\n".join(lines),
                ha="left", va="top", fontsize=bs, color=INK,
                linespacing=G1_LS, zorder=3)
    if note:
        ax.text(x + w / 2, y_top - h + pad, note, ha="center", va="bottom",
                fontsize=bs, color="#52514e", style="italic", zorder=3)
    return dict(L=x, R=x + w, T=y_top, B=y_top - h, x=x + w / 2)


def _g1_arrow(ax, verts, lw, color, head=True):
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    path = Path(verts, [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>" if head else "-",
                                 mutation_scale=9 + 3 * (lw > 2), linewidth=lw,
                                 color=color, shrinkA=0, shrinkB=0,
                                 joinstyle="miter", zorder=1))


# 상자 내용 — 명세의 문구에 누적 교정과 10-28 갱신을 반영한 것
G1_B1 = ("데이터 수집", ["DeepHF 공개 데이터셋 (Wang et al. 2019)",
                    "HEK293T 세포 · 합성 표적 라이브러리",
                    "효소 3종 측정값 · 공통 20-mer 53,937개"])
G1_B2 = ("목적함수 정의", [r"$Y_1$ — 절단 효율 (야생형)",
                     r"$Y_2$ — 활성 유지율 (SpCas9-HF1 ÷ 야생형)",
                     r"$Y_{2v}$ — eSpCas9(1.1), 검증 전용"])
G1_B2_NOTE = "이후 단계에서 두 목적함수의 관계를 분석"
G1_B3 = ("분위 스캔", ["상위 분위 q 0.30 → 0.03",
                   "두 목적 동시 만족 가이드 수 집계",
                   "전체 53,937 · 평가셋 5,394 두 규모",
                   "가우시안 코퓰러 귀무분포와 대조",
                   "해석해 교차확인 · 교란 4종 배제"])
G1_CHAIN = [
    ("데이터 분할", ["서열 유사 가이드를 한 덩어리로 (Hamming ≤ 3)",
                 "학습 43,149 / 검증 5,394 / 평가 5,394"]),
    ("멀티태스크 CNN 학습",
     [r"20-mer 원핫 + 스칼라 11개 · 공유층 → 2개 head ($\hat{Y}_1$, $\hat{Y}_2$)",
      # 예측값에 반드시 햇을 붙인다 — MSE(Y₁, Y₁) = 0 이라 햇이 없으면 식이 항등적으로 0이다
      r"$L = \lambda \cdot \mathrm{MSE}(\hat{Y}_1,\, Y_1)"
      r" + (1-\lambda) \cdot \mathrm{MSE}(\hat{Y}_2,\, Y_2)$",
      "가중치 λ 0 → 1 스윕 (0.1 간격 11개)"]),
    ("파레토 전선 산출", ["비지배 정렬 (동점은 평균순위)",
                   "귀무분포 · 잡음 교란과 대조"]),
    ("서열 해석", ["Saliency map", "가상 돌연변이 시뮬레이션"]),
    ("독립 검증 — 네 가지를 나란히 수행",
     [r"미학습 효소 eSpCas9 전이 ($Y_{2v}$) · 예측 → 원자료 대조",
      "베이스라인 대조 · 사전 등록 대조"]),
]
G1_B9 = ("재현성 확인",
         ["난수 시드 고정 · 시드 5개 반복 (산포는 표본표준편차, ddof = 1)",
          "연산 결정성 on/off 대조 · 측정값 경로 재학습 영향 없음 / "
          "예측값 경로 재학습 시 변동"])


def fig0_study_design(height_mm=None):
    """본문 그림 1 — 연구 설계 개요.

    세로 길이를 내용에서 역산한다 (height_mm 을 주면 그 값으로 고정).
    상자 안 수치는 데이터 규모뿐이며 분석 결과가 아니다 (명세 규칙 1).
    """
    MEAS, PRED = INK, "#8a8985"              # 측정값 경로 / 예측값 경로
    LW_M, LW_P = 2.6, 1.2
    F_COM, F_MEAS, F_PRED = "#fcfcfb", "#cde2fb", "#f0efec"
    M, GAP_BR, GAP_ROW, GAP_CH, GAP_BAND = 6.0, 15.0, 17.0, 7.0, 12.0   # 모두 pt

    # ---- 1단계 : 세로 길이를 pt 로 먼저 합산한다
    h1 = max(_g1_h(G1_B1[1]), _g1_h(G1_B2[1], note=True))
    h3 = _g1_h(G1_B3[1])
    hc = [_g1_h(ls) for _, ls in G1_CHAIN]
    h9 = _g1_h(G1_B9[1])
    right = sum(hc) + GAP_CH * (len(hc) - 1)
    H_pt = M + h1 + GAP_BR + GAP_ROW + max(h3, right) + GAP_BAND + h9 + M
    if height_mm:
        H_pt = height_mm / 25.4 * 72.0

    fig, ax = plt.subplots(figsize=(G1_W_MM / 25.4, H_pt / 72.0), layout="constrained")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor("#fcfcfb")
    Y = lambda pt: 1.0 - pt / H_pt           # pt(위에서부터) → 축 좌표

    b1 = _g1_box(ax, 0.015, Y(M), 0.455, *G1_B1, F_COM, INK, 1.4, H_pt)
    b2 = _g1_box(ax, 0.530, Y(M), 0.455, *G1_B2, F_COM, INK, 1.4, H_pt,
                 note=G1_B2_NOTE)
    _g1_arrow(ax, [(b1["R"], Y(M + h1 / 2)), (b2["L"], Y(M + h1 / 2))], 1.4, INK)

    y_j = Y(M + h1 + GAP_BR)
    y_row = Y(M + h1 + GAP_BR + GAP_ROW)
    xm, xp = 0.185, 0.6975
    _g1_arrow(ax, [(b2["x"], b2["B"]), (b2["x"], y_j)], 1.4, INK, head=False)
    ax.text(b2["x"] - 0.012, y_j + 2.0 / H_pt, "모델을 거치는가", ha="right",
            va="bottom", fontsize=G1_BS, color="#52514e")

    _g1_arrow(ax, [(b2["x"], y_j), (xm, y_j), (xm, y_row)], LW_M, MEAS)
    ax.text(0.40, y_j + 2.0 / H_pt, "측정값만 사용 — 신경망을 거치지 않음",
            ha="center", va="bottom", fontsize=G1_BS, color=MEAS,
            fontweight="bold")
    b3 = _g1_box(ax, 0.015, y_row, 0.340, *G1_B3, F_MEAS, MEAS, 2.4, H_pt)

    _g1_arrow(ax, [(b2["x"], y_j), (xp, y_j), (xp, y_row)], LW_P, PRED)
    ax.text(xp + 0.008, (y_j + y_row) / 2, "신경망 예측값 사용", ha="left",
            va="center", fontsize=G1_BS, color="#52514e")

    RX, RW = 0.420, 0.565
    top_pt = M + h1 + GAP_BR + GAP_ROW
    boxes = []
    for i, (t, ls) in enumerate(G1_CHAIN):
        b = _g1_box(ax, RX, Y(top_pt), RW, t, ls, F_PRED, PRED, LW_P, H_pt)
        boxes.append(b)
        if i:
            _g1_arrow(ax, [(b["x"], boxes[i - 1]["B"]), (b["x"], b["T"])], LW_P, PRED)
        top_pt += hc[i] + GAP_CH

    # 재현성 띠 — 두 경로에 걸친다. 특정 상자를 가리키는 화살표를 붙이지 않는다
    y_band = Y(M + h1 + GAP_BR + GAP_ROW + max(h3, right) + GAP_BAND)
    _g1_box(ax, 0.015, y_band, 0.970, *G1_B9, F_PRED, "#c9c8c2", 1.0, H_pt)

    fig.canvas.draw()
    inv = ax.transData.inverted()
    over = []
    for pt in ax.patches:
        if type(pt).__name__ != "FancyBboxPatch":
            continue
        bb = pt.get_window_extent()
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        if x0 < -1e-6 or x1 > 1 + 1e-6 or y0 < -1e-6 or y1 > 1 + 1e-6:
            over.append((round(x0, 3), round(x1, 3), round(y0, 3), round(y1, 3)))
    if over:
        raise SystemExit(f"[중단] 상자가 그림 밖으로 나감: {over}")

    save(fig, "fig0_study_design", dpi=400)
    return dict(width_mm=G1_W_MM, height_mm=round(H_pt / 72.0 * 25.4, 1))


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
    step("front_null", fig_front_null)
    step("fig0", fig0_study_design)

    # --- 파이프라인 개요 (번호 없음)
    step("pipeline", fig_pipeline_overview)
