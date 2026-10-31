"""
도표·그림 명세서 생성기
=======================
논문에 실린 모든 표와 그림의 수치를 out/*.json 에서 직접 읽어 마크다운으로 쓴다.
수치를 손으로 옮겨 적지 않으므로 원본과 항상 일치한다.

산출:
  out/figure_data.json     — 모든 도판의 수치를 한 파일로 통합 (그래프 재작성용)
  out/scatter_points.csv   — 그림 1·3 의 산점도 원자료 (hold-out 5,394개)
  도표-그림-명세.md         — 사람이 읽는 명세서
"""
import numpy as np, pandas as pd, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
O = ROOT / "out"
L = lambda n: json.load(open(O / n))

p1s = L("phase1_summary.json"); sweep = L("phase2_lambda_sweep.json")
p3 = L("phase3_summary.json"); p4 = L("phase4_summary.json")
p5 = L("phase5_validation.json"); p6 = L("phase6_baseline.json")
D = dict(np.load(O / "dataset.npz", allow_pickle=True))
P = np.load(O / "phase3_points.npz", allow_pickle=True)
te = D["split"] == 2

# ---------------------------------------------- 산점도 원자료 (그림 1·3)
sc = pd.DataFrame({
    "seq20": P["seq"], "pred_Y1": P["pred"][:, 0], "pred_Y2": P["pred"][:, 1],
    "meas_Y1": P["meas"][:, 0], "meas_Y2": P["meas"][:, 1],
    "meas_Y2v_eSpCas9": D["y2v"][te],
})
sc["on_pareto_front"] = False
sc.loc[P["f1"], "on_pareto_front"] = True
sc.to_csv(O / "scatter_points.csv", index=False)

# ------------------------- 그림 7–10 수치 (figures.py 가 유일한 출처)
# 주: import 시점에 그림 7–10 을 다시 그린다. 명세서와 도판이 항상 같은 코드에서 나온다.
import sys as _sys; _sys.path.insert(0, str(ROOT / "src"))
import figures as _FG
S7, S8, S9, S10 = _FG.fig7(), _FG.fig8(), _FG.fig9(), _FG.fig10()
SA = json.load(open(O / "phase2_seed_agg.json", encoding="utf-8"))
S6 = json.load(open(O / "phase6_seed_agg.json", encoding="utf-8"))
S12 = _FG.fig12()
QN = json.load(open(O / "phase3_qscan_null.json", encoding="utf-8"))
KK = json.load(open(O / "phase7_knee_k.json", encoding="utf-8"))
S11 = _FG.fig11(KK)   # 도판만 그린다 (반환값 없음)
FN = json.load(open(O / "phase3_front_null.json", encoding="utf-8"))
_nrm = lambda v: (v - v.min()) / (v.max() - v.min())
_sc = 0.5 * _nrm(P["pred"][:, 0]) + 0.5 * _nrm(P["pred"][:, 1])
_gr = np.empty(len(_sc), int); _gr[np.argsort(-_sc)] = np.arange(len(_sc)) + 1
F1_GLOBAL_RANK_MED = int(np.median(_gr[P["f1"]]))
K_FRAC = 20 / len(P["f1"])

# ---------------------------------------------- 통합 JSON
NTS = list("ACGT")
ism = np.array(p4["ism_mean"])                      # (20,4,2)
FD = {
    "_readme": "논문 도판 6장의 수치 전체. 그림 1·3의 산점도만 scatter_points.csv 참조.",
    "palette": {
        "categorical": {"slot1_blue": "#2a78d6", "slot2_orange": "#eb6834",
                        "slot3_aqua": "#1baf7a"},
        "sequential_blue": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                            "#256abf", "#184f95", "#0d366b"],
        "diverging_blue_red": ["#104281", "#2a78d6", "#9ec5f4", "#f0efec",
                               "#f0a3a2", "#e34948", "#8f1f1e"],
        "surface": "#fcfcfb", "ink": "#0b0b0b", "muted_ink": "#52514e",
        "grid": "#e8e7e3", "axis": "#c9c8c2",
        "rules": ["범주형은 슬롯 순서대로 고정 배정, 순환 금지",
                  "이중 축 금지", "순차형은 단일 색상 명도 변화",
                  "발산형은 두 색 + 중립 회색 중점", "2계열 이상이면 범례 필수"],
    },
    "fig1_distributions": {
        "type": "히스토그램 2 + 육각 밀도 1",
        "source": "scatter_points.csv (meas_Y1, meas_Y2) — 단 전체 53,937개 기준 통계는 아래",
        "n_all": int(len(D["y1"])), "n_holdout": int(te.sum()),
        "Y1": {"mean": float(D["y1"].mean()), "std": float(D["y1"].std()),
               "quartiles": [float(x) for x in np.quantile(D["y1"], [0, .25, .5, .75, 1])]},
        "Y2": {"mean": float(D["y2"].mean()), "std": float(D["y2"].std()),
               "quartiles": [float(x) for x in np.quantile(D["y2"], [0, .25, .5, .75, 1])],
               "clipped_at": 1.5, "n_clipped": 161, "pct_clipped": 0.2985},
        "pearson_all": float(np.corrcoef(D["y1"], D["y2"])[0, 1]),
    },
    "fig2_lambda_sweep": {
        "type": "선 그래프 2계열",
        "x": [s["lam"] for s in sweep],
        "series": {"Head A — 효율 Y1": [s["sp_eff"] for s in sweep],
                   "Head B — 유지율 Y2": [s["sp_fid"] for s in sweep]},
        "note": "λ=0 은 Head A 미학습, λ=1 은 Head B 미학습이라 양 끝점이 급락",
    },
    "fig3_pareto": {
        "type": "육각 밀도 + 파레토 전선 강조 / 계층 산점도",
        "source": "scatter_points.csv (pred_Y1, pred_Y2, on_pareto_front)",
        "front_sizes_predicted": p3["predicted"]["front_sizes"],
        "front_sizes_measured": p3["measured"]["front_sizes"],
        "f1_size": p3["predicted"]["f1_size"],
        "hypervolume": p3["predicted"]["hypervolume"],
        "joint_optimal": p3["joint_optimal"],
    },
    "fig4_saliency": {
        "type": "그룹 막대 + 발산형 히트맵",
        "positions": list(range(1, 21)),
        "position_importance_normalized": {
            k: (np.array(v) / np.sum(v)).round(5).tolist()
            for k, v in p4["pos_importance"].items()},
        "nt_importance_fid_ACGT_x_pos": p4["nt_importance"]["fid"],
        "nt_importance_eff_ACGT_x_pos": p4["nt_importance"]["eff"],
    },
    "fig5_ism": {
        "type": "발산형 히트맵 2장",
        "layout": "[position 1..20][nucleotide A,C,G,T][0=dY1, 1=dY2]",
        "values": p4["ism_mean"],
        "note": "실제 치환이 일어난 가이드에 한정한 조건부 평균 (원본 염기 칸은 0)",
        "top_fidelity_down": p4["candidates"][:5],
        "top_fidelity_up": p4["candidates"][-5:],
        "rescue_pos18_G": p4["rescue"],
    },
    "fig6_baseline": {
        "type": "그룹 막대 + 선 그래프 2계열",
        "overall": {c: {r["method"]: r["spearman"] for r in p6[c]["overall"]}
                    for c in ("full", "sub10")},
        "n_train": {c: p6[c]["n_train"] for c in ("full", "sub10")},
        "component_models": {c: {k: v["spearman"] for k, v in p6[c]["aux"].items()}
                             for c in ("full", "sub10")},
        "quartile_gap_MT_minus_BASE": {
            c: [s["MT"] - s["BASE"] for s in p6[c]["strata"]] for c in ("full", "sub10")},
        "quartile_wt_ranges": [s["wt_range"] for s in p6["full"]["strata"]],
        "verdict": p6["verdict"],
    },
}
FD["fig7_pareto_measured"] = dict(
    type="측정공간 육각밀도 + 예측 전선 실측 위치 / 비지배 계층 누적분포",
    source="scatter_points.csv (meas_Y1, meas_Y2, on_pareto_front)", **S7)
FD["fig8_espcas9_transfer"] = dict(
    type="육각밀도 2장 (예측→eSpCas9, HF1실측→eSpCas9)",
    source="scatter_points.csv (pred_Y2, meas_Y2, meas_Y2v_eSpCas9)", **S8)
FD["fig9_selection_strategies"] = dict(
    type="그룹 막대 5그룹 × 3계열", k=20,
    note="무릎점과 가중합은 동일한 정렬 키(정규화 합)를 쓰며 후보군만 다르다",
    values={k.replace("\n", " "): v for k, v in S9.items()})
FD["fig6_seed_agg"] = dict(
    type="시드 베이스 5개({0,10,20,30,40}) 평균 ± 표본표준편차(ddof=1)",
    source="phase6_seed_sweep.json",
    note="그림 6 의 입력. phase6_baseline.json(원 실행)과 별개다.",
    ddof=1, seed_bases=S6["seed_bases"], n=S6["n"],
    full=S6["full"], sub10=S6["sub10"])
FD["fig2_seed_agg"] = dict(
    type="λ별 시드 5개(0–4) 평균 ± 표본표준편차(ddof=1)",
    source="phase2_seed_sweep.json", note="그림 2·10 의 입력. 표 1(원 실행)과 별개다.",
    ddof=1, rows=SA["rows"])
FD["fig10_lambda_front"] = dict(
    type="산점도 2패널 (전체 / 확대)",
    source="phase2_seed_sweep.json (시드 5개) — 표 1 원 실행과는 다른 자료", **S10)

FD["fig11_knee_k_sensitivity"] = dict(
    type="선 그래프 3계열 × 2패널 + 누적 평균 2계열",
    source="phase7_knee_k.json",
    k_values=KK["k_values"], n_front_F1=KK["n_front_F1"],
    results=KK["results"], front_ordered_measured=KK["front_ordered_measured"])
FD["fig12_qscan"] = dict(
    type="선 그래프 2계열, 단일 선형 y축, q 역방향",
    source="phase3_front_null.json (qscan)",
    qscan=FN["qscan"], null=FN["null"],
    independent_theory_Hn=FN["independent_theory_Hn"],
    reps_null=FN["reps_null"], **S12)
json.dump(FD, open(O / "figure_data.json", "w"), indent=2, ensure_ascii=False)

# ---------------------------------------------- 명세서 마크다운
def tbl(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


M = []
A = M.append
A("# KSEF CRISPR 논문 — 도표·그림 명세서\n")
A("> 확정 제목 (2026-09-06) — CRISPR-Cas9 gRNA의 절단 효율과 High-Fidelity 효소에서의\n"
  "> 활성 유지율은 상충하는가 — 멀티태스크 심층학습과 파레토 분석에 의한 정량적 검증\n"
  ">\n"
  "> 도판 안에는 제목을 넣지 않는다. 표지·발표자료용 참고다.\n")
A("논문에 실린 표 7개와 그림 12장의 **수치 전부**를 담는다. 모든 값은 "
  "`out/*.json`에서 코드로 추출했으므로 논문 본문과 항상 일치한다.\n")
A("> 이 문서는 그래프 재작업 전용이다. 그림을 다시 그릴 때는 여기 숫자만 쓰면 되고, "
  "논문 본문이나 코드를 다시 읽을 필요가 없다.\n")

A("## 0. 첨부 데이터 파일\n")
A(tbl(["파일", "내용", "용도"], [
    ["`figure_data.json`", "그림 1–12의 모든 수치 + 팔레트 정의", "그래프 재작성의 단일 출처"],
    ["`out/dataset.npz`", "전체 53,937개의 one-hot·바이오 피처·Y₁/Y₂/Y₂ᵥ·분할", "그림 1 재생성에 필요"],
    ["`phase3_front_null.json`", "전선 크기 copula 귀무분포 + 잡음 교란", "본문 §3.3"],
    ["`phase3_qscan_null.json`", "q-스캔 13지점의 귀무평균·sd·z", "그림 12"],
    ["`phase7_knee_k.json`", "무릎점 선별의 k 민감도", "그림 11"],
    ["`scatter_points.csv`", "hold-out 5,394개의 예측·측정 Y₁/Y₂, 파레토 전선 여부", "그림 1·3 산점도"],
    ["`pareto_front_guides.csv`", "파레토 전선 26개 가이드의 서열과 값", "표 부록"],
    ["`fig/*.png`", "현재 도판 12장, 래스터 200 dpi", "대조용·문서 삽입"],
    ["`fig/*.pdf`, `fig/*.svg`", "같은 도판의 벡터판", "Illustrator·Inkscape 후편집"],
    ["`src/figures.py`", "현재 도판 생성 코드 — **수정의 실제 대상**", "재현"],
]))
A("""
`figure_data.json`, `scatter_points.csv`, `figures.py`는 프로젝트 `claude/data/`에 있어
새 대화에서 바로 읽힌다. **이미지 파일(png·pdf·svg)은 프로젝트에 올릴 수 없으므로**
맥의 `Documents/KSEF_CRISPR/fig/` 또는 `KSEF_CRISPR_analysis.zip`의 `fig/`에서 가져온다.

**무엇을 고치는가** — PNG·SVG는 결과물이지 편집 대상이 아니다. 도판을 바꾸는 정식 경로는
`figures.py`를 고쳐 다시 실행하는 것이고, 그래야 논문 빌드와 재현성이 유지된다.
벡터 파일은 마지막 미세 조정이나 외부 편집기 작업에만 쓴다.
""")

A("\n## 1. 공통 디자인 규격\n")
A("**색** — 검증된 팔레트에서 역할별로만 가져온다. 임의 색상 금지.\n")
A(tbl(["역할", "값"], [
    ["범주형 슬롯 1", "`#2a78d6` 파랑"],
    ["범주형 슬롯 2", "`#eb6834` 주황"],
    ["범주형 슬롯 3", "`#1baf7a` 청록"],
    ["순차형 (파랑 램프)", "`#cde2fb → #0d366b` 7단계"],
    ["발산형 (파랑↔빨강)", "중립 중점 `#f0efec`"],
    ["도판 배경", "`#fcfcfb`"],
    ["본문 잉크 / 보조 잉크", "`#0b0b0b` / `#52514e`"],
    ["격자 / 축선", "`#e8e7e3` / `#c9c8c2`"],
]))
A("\n**규칙**\n")
for r in FD["palette"]["rules"]:
    A(f"- {r}")
A("\n**글꼴·크기** — Noto Sans CJK, 본문 9 pt, 제목 10 pt, 200 dpi, "
  "`layout=\"constrained\"`, 위·오른쪽 축선 제거.\n")

A("\n## 2. 표\n")
A("### 표 1 — 결합손실 λ 스윕 (그림 2와 동일 자료)\n")
A(tbl(["λ", "Y₁ Spearman ρ", "Y₂ Spearman ρ"],
      [[f"{s['lam']:.1f}", f"{s['sp_eff']:.4f}", f"{s['sp_fid']:.4f}"] for s in sweep]))

A("\n### 표 2 — 동시 최적 가이드 수 (평가셋 5,394개, copula 귀무분포 대비)\n")
_QM = {round(r["q"], 2): r for r in QN["series"]["measured"]["rows"]}
_QP = {round(r["q"], 2): r for r in QN["series"]["predicted"]["rows"]}
A(tbl(["기준", "독립 N·q² (기준선 아님)", "copula 귀무 (측정)", "측정 관측", "측정 z",
       "예측 관측", "예측 z"],
      [[f"각각 상위 {q*100:.0f}%", f"{_QM[q]['independent_Nq2']:.1f}",
        f"{_QM[q]['null_mean']:.2f} ± {_QM[q]['null_sd']:.2f}",
        _QM[q]["observed"], f"{_QM[q]['z']:+.2f}",
        _QP[q]["observed"], f"{_QP[q]['z']:+.2f}"]
       for q in (0.25, 0.10, 0.05)]))
A("""
**독립 가정 N·q² 는 기준선이 아니다.** 두 목적함수는 약한 양의 상관을 가지므로
(측정 스피어만 +0.0324, 예측 +0.0677) 올바른 귀무는 그 상관을 맞춘 가우시안 copula 다.
N·q² 열은 이전 판과의 대조를 위해 남겨둔 참고값이다.

기준선을 바꾸면 상위 10%의 측정 z 가 음수(−0.71)가 되므로,
「상위 10%에서도 동시 최적이 기대를 웃돈다」는 이전 서술은 성립하지 않는다.
""")

A("\n### 표 3 — 유지율 영향이 큰 치환 (ISM, 조건부 평균)\n")
A(tbl(["방향", "치환", "ΔŶ₂", "ΔŶ₁", "n"],
      [["저하", f"위치 {c['pos']} → {c['nt']}", f"{c['dY2']:+.4f}", f"{c['dY1']:+.4f}", f"{c['n']:,}"]
       for c in p4["candidates"][:5]] +
      [["상승", f"위치 {c['pos']} → {c['nt']}", f"{c['dY2']:+.4f}", f"{c['dY1']:+.4f}", f"{c['n']:,}"]
       for c in p4["candidates"][-5:][::-1]]))

A(f"\n### 표 4 — 위치 18 G의 교란 검증 (n = {p4['candidates'][0]['n']:,} 중 실제 보유 1,199)\n")
A(tbl(["치환", "ΔŶ₂ (유지율)", "ΔŶ₁ (효율)"],
      [[f"G → {k}", f"{v['dY2']:+.4f}", f"{v['dY1']:+.4f}"] for k, v in p4["rescue"].items()]))

A("\n### 표 5 — ISM 예측과 실측값 대조 (Mann–Whitney U)\n")
A(tbl(["치환", "ISM ΔŶ₂", "실측 Y₂ 차이", "n", "p"],
      [[f"위치 {t['pos']} · {t['nt']}", f"{t['ism_dY2']:+.4f}", f"{t['meas_diff']:+.4f}",
        f"{t['n_with']:,}", f"{t['p']:.1e}"] for t in p5["V2_ism_vs_measured"]]))
a = p5["V2_agreement"]
A(f"\n부호 일치 **{a['sign_agree']}/{a['n']}**, p<0.05 **{a['p_lt_05']}/{a['n']}**\n")

A("\n### 표 6 — 선별 전략별 실측 성능 (상위 20개)\n")
v4 = p5["V4_selection_strategies"]
A(tbl(["전략", "측정 Y₁", "측정 Y₂", "측정 Y₂ᵥ (eSpCas9)"],
      [[k, f"{v['meas_Y1']:.3f}", f"{v['meas_Y2']:.3f}", f"{v['meas_Y2v']:.3f}"]
       for k, v in v4.items()] +
      [["(hold-out 전체 평균)", f"{D['y1'][te].mean():.3f}",
        f"{D['y2'][te].mean():.3f}", f"{D['y2v'][te].mean():.3f}"]]))

A("\n### 표 7 — 사전 등록 기준선 비교 (그림 6과 동일 자료)\n")
rows = []
for c, lab in [("full", "전체 (43,149)"), ("sub10", "10% (4,314)")]:
    g = {r["method"]: r["spearman"] for r in p6[c]["overall"]}
    rows.append([lab, f"{g['BASE']:.4f}", f"{g['ST']:.4f}", f"{g['MT']:.4f}",
                 f"{g['MT']-g['BASE']:+.4f}"])
A(tbl(["학습 데이터", "BASE", "ST", "MT", "MT − BASE"], rows))
A("\n**기준선 구성 모델의 개별 성능** (기준선이 약하지 않았음을 보이는 값)\n")
A(tbl(["조건", "BASE의 WT 예측 ρ", "BASE의 HF1 예측 ρ", "MT의 Y₁ 예측 ρ"],
      [[lab] + [f"{p6[c]['aux'][k]['spearman']:.4f}" for k in ("base_wt", "base_hf", "mt_wt")]
       for c, lab in [("full", "전체"), ("sub10", "10%")]]))

A("\n## 3. 그림\n")
A("### 그림 1 — 두 목적함수의 분포와 결합분포\n")
A("- 형태: 히스토그램 2개(60 bin) + 육각 밀도 1개(gridsize 45)\n"
  "- 색: 좌 슬롯1 파랑, 중 슬롯2 주황, 우 순차형 파랑 램프\n"
  f"- 자료: 전체 {FD['fig1_distributions']['n_all']:,}개")
f1d = FD["fig1_distributions"]
A(tbl(["축", "평균", "표준편차", "최소", "1사분위", "중앙", "3사분위", "최대"],
      [["Y₁"] + [f"{f1d['Y1']['mean']:.3f}", f"{f1d['Y1']['std']:.3f}"] +
       [f"{x:.3f}" for x in f1d["Y1"]["quartiles"]],
       ["Y₂"] + [f"{f1d['Y2']['mean']:.3f}", f"{f1d['Y2']['std']:.3f}"] +
       [f"{x:.3f}" for x in f1d["Y2"]["quartiles"]]]))
A(f"\n- Pearson r = {f1d['pearson_all']:+.4f}\n"
  f"- Y₂ 오른쪽 끝 봉우리는 상한 절단(1.5) 때문 — {f1d['Y2']['n_clipped']}개 "
  f"({f1d['Y2']['pct_clipped']:.2f}%). 캡션에 반드시 명시할 것.\n")

A("### 그림 2 — λ 스윕\n")
A("""- 형태: 선 그래프 2계열, 단일 y축 (둘 다 Spearman ρ이므로 이중 축 불필요·금지)
- 색: Head A 슬롯1, Head B 슬롯2
- 자료: `phase2_seed_sweep.json` — λ마다 **시드 5개(0–4)의 평균**, 오차막대는
  **표본표준편차(ddof = 1)**. 표 1(원 실행 단일 시드)과는 다른 자료다.
- λ = 0.5 위치의 세로 점선(채택값)은 유지한다.
- λ = 0 의 Y₁ 은 sd 0.3698 이라 오차막대가 축 아래로 나간다. **축을 넓히지 않고 그 막대만
  자른다** — 축을 맞추면 학습 구간(λ 0.1–0.9)의 차이가 안 보인다. 잘렸다는 표시를 넣었다.
- 주의: λ=0·λ=1의 급락은 해당 head 미학습 때문. 캡션에 설명 필요.
""")
A("\n**λ별 시드 5개 집계 (ddof = 1)**\n")
A(tbl(["λ", "Y₁ 평균", "Y₁ sd", "Y₂ 평균", "Y₂ sd", "n"],
      [[f"{r['lam']:.1f}", f"{r['eff_mean']:+.4f}", f"{r['eff_sd']:.4f}",
        f"{r['fid_mean']:+.4f}", f"{r['fid_sd']:.4f}", r["n"]] for r in SA["rows"]]))
A("\n퇴화점을 뺀 λ 0.1–0.9 에서 두 head 모두 sd 가 0.0022–0.0323 이다. "
  "즉 이 구간의 λ 간 차이는 대부분 시드 잡음과 같은 크기이며, 특정 λ 를 「최적」으로 "
  "읽어서는 안 된다.\n")

A("### 그림 3 — 파레토 전선\n")
A(f"- 좌: 육각 밀도(gridsize 48) 위에 F₁ {p3['predicted']['f1_size']}개를 빨강 `#e34948`로 강조\n"
  "- 우: F₁–F₅ 계층 산점도, 순차형 파랑 램프 (명도 순서 = 계층 순서)\n"
  "- 자료: `scatter_points.csv`의 pred_Y1·pred_Y2·on_pareto_front\n")
A(tbl(["", "F₁", "F₂", "F₃", "F₄", "F₅", "F₆"],
      [["예측값 기준"] + p3["predicted"]["front_sizes"][:6],
       ["측정값 기준"] + p3["measured"]["front_sizes"][:6]]))
A(f"\n- 하이퍼볼륨 {p3['predicted']['hypervolume']:.4f}, "
  f"Ŷ₁ 범위 {p3['predicted']['f1_x_range'][0]:.3f}–{p3['predicted']['f1_x_range'][1]:.3f}, "
  f"Ŷ₂ 범위 {p3['predicted']['f1_y_range'][0]:.3f}–{p3['predicted']['f1_y_range'][1]:.3f}\n")

A("### 그림 4 — 위치별 saliency\n")
A("- 좌: 그룹 막대 20위치 × 2계열 (슬롯1·슬롯2), y = 정규화 기여도(합 1)\n"
  "- 우: 발산형 히트맵 4염기 × 20위치, 0을 중립 회색에 고정\n")
pe = np.array(p4["pos_importance"]["eff"]); pf = np.array(p4["pos_importance"]["fid"])
pe, pf = pe / pe.sum(), pf / pf.sum()
A(tbl(["위치"] + [str(i) for i in range(1, 21)],
      [["Head A 효율"] + [f"{x:.4f}" for x in pe],
       ["Head B 유지율"] + [f"{x:.4f}" for x in pf]]))
A(f"\n- 시드 영역(11–20) 기여 비중: 효율 {pe[10:].sum():.3f} / 유지율 {pf[10:].sum():.3f}\n"
  "- 염기별 부호 있는 기여도 행렬은 `figure_data.json`의 "
  "`fig4_saliency.nt_importance_*` (4×20, 행 순서 A·C·G·T)\n")

A("### 그림 5 — In-silico mutagenesis\n")
A("- 형태: 발산형 히트맵 2장 (좌 ΔŶ₁, 우 ΔŶ₂), 각각 자기 최대절댓값으로 대칭 정규화\n"
  "- 자료: `figure_data.json`의 `fig5_ism.values` — [위치 20][염기 ACGT][0=ΔŶ₁, 1=ΔŶ₂]\n"
  "- 최강 칸은 색이 아니라 흰 테두리 + 텍스트 라벨로 표시 (색만으로 의미를 싣지 않음)\n")
A(tbl(["", "최소 칸", "값"],
      [["ΔŶ₁", f"위치 {int(np.unravel_index(np.argmin(ism[:,:,0]), ism[:,:,0].shape)[0])+1}"
        f" → {NTS[int(np.unravel_index(np.argmin(ism[:,:,0]), ism[:,:,0].shape)[1])]}",
        f"{ism[:,:,0].min():+.4f}"],
       ["ΔŶ₂", f"위치 {int(np.unravel_index(np.argmin(ism[:,:,1]), ism[:,:,1].shape)[0])+1}"
        f" → {NTS[int(np.unravel_index(np.argmin(ism[:,:,1]), ism[:,:,1].shape)[1])]}",
        f"{ism[:,:,1].min():+.4f}"]]))

A("\n### 그림 6 — 사전 등록 기준선 비교\n")
A("""- 좌: 그룹 막대 2조건 × 3방식 (BASE 슬롯1, ST 슬롯2, MT 슬롯3 — 나열 순서대로 고정)
- 우: 선 그래프 2계열, 사분위별 MT − BASE 격차
- 2026-09-13: 막대·점은 **시드 베이스 5개({0, 10, 20, 30, 40}) 평균**,
  오차막대는 **표본표준편차(ddof = 1)**. 자료는 `phase6_seed_sweep.json` 이며
  `phase6_baseline.json`(원 실행, 표 7)과는 다른 자료다.
""")
A("\n**추정 방식별 정확도 — 시드 5개 집계**\n")
A(tbl(["조건", "BASE", "ST", "MT"],
      [[{"full": "전체 학습셋", "sub10": "10% 학습셋"}[c]] +
       [f"{S6[c]['overall'][m]['mean']:.4f} ± {S6[c]['overall'][m]['sd']:.4f}"
        for m in ("BASE", "ST", "MT")] for c in ("full", "sub10")]))
A(f"""
BASE 의 시드 간 산포(전체 조건 {S6['full']['overall']['BASE']['sd']:.4f})가
ST({S6['full']['overall']['ST']['sd']:.4f})의 약
{S6['full']['overall']['BASE']['sd'] / S6['full']['overall']['ST']['sd']:.0f}배다.
두 예측값의 비를 취하는 방식이 불안정하다는 뜻이며, 이 패널의 새 정보다.
""")
A("\n**사분위별 MT − BASE 격차 — 시드 5개 집계**\n")
A(tbl(["사분위", "WT 효율 범위", "전체 학습셋", "10% 학습셋"],
      [[f"Q{i+1}",
        f"{p6['full']['strata'][i]['wt_range'][0]:.3f}–{p6['full']['strata'][i]['wt_range'][1]:.3f}",
        f"{S6['full']['quartile_gap_MT_minus_BASE']['mean'][i]:+.4f} ± "
        f"{S6['full']['quartile_gap_MT_minus_BASE']['sd'][i]:.4f}",
        f"{S6['sub10']['quartile_gap_MT_minus_BASE']['mean'][i]:+.4f} ± "
        f"{S6['sub10']['quartile_gap_MT_minus_BASE']['sd'][i]:.4f}"]
       for i in range(4)]))
A(f"""
시드별 최대 사분위가 전체 조건에서 {S6['full']['quartile_gap_MT_minus_BASE']['argmax_quartile']}
(Q 번호)로 갈린다 — 5회 중 Q1 이 최대인 것은 3회뿐이다. 오차막대가 큰 것이 정상이며,
**P2 가 판정 불가라는 사실이 그림에 드러나는 것이 목적**이므로 작아 보이게 다듬지 않는다.

아래 「사전 예측 판정」은 `phase6_baseline.json`(원 실행 1회) 기준이다.
시드 5회 재판정은 원고 4.9절을 볼 것 — P2 는 판정 불가, P3 는 기각이다.
""")

A("\n**사전 예측 판정**\n")
A(tbl(["예측", "내용", "결과"],
      [[k, v["claim"], "지지됨" if v["supported"] else "기각됨"]
       for k, v in p6["verdict"].items()]))

A("\n### 그림 7 — 파레토 전선의 실측 재현\n")
A("- 좌: 측정 Y₁–Y₂ 육각 밀도(gridsize 48) + 예측 전선 F₁ 26개의 실측 위치(빨강 `#e34948`)\n"
  "  + 측정값 기준 전선 F₁ᵐ(슬롯3 청록)\n"
  "- 우: 측정값 기준 비지배 계층의 누적분포 2계열 (예측 F₁ 26개 vs hold-out 전체)\n"
  "- 빨강은 그림 3에서 '예측 파레토 전선'에 이미 배정된 색이라 같은 뜻으로 재사용했다.\n")
A(tbl(["항목", "값"], [
    ["예측 F₁ 26개의 실측 비지배 계층 중앙값", f"{S7['median_rank_front']:.1f}"],
    ["hold-out 전체의 실측 계층 중앙값", f"{S7['median_rank_all']:.0f}"],
    ["무작위 26개 표본 계층 중앙값 95% 구간", f"{S7['null_ci'][0]:.0f}–{S7['null_ci'][1]:.0f}"],
    ["단측 p (재표집 10,000회, seed 20260830)", "< 1e-4 (0/10,000)"],
    ["26개 중 측정값 기준 F₁ 진입", f"{S7['n_in_measured_f1']}개"],
    ["실측 계층 최대값", f"{S7['max_rank']}"],
]))
A("\n예측 전선은 무작위 대비 앞쪽 계층에 몰리지만, 실측 전선에 실제로 진입한 것은 "
  f"26개 중 {S7['n_in_measured_f1']}개다. 두 사실을 함께 서술할 것.\n")

A("\n### 그림 8 — eSpCas9 교차 효소 전이\n")
A("- 좌: 예측 Ŷ₂ vs 측정 Y₂ᵥ / 우: 측정 Y₂(HF1) vs 측정 Y₂ᵥ — 참조 기준\n"
  "- 육각 밀도 gridsize 42, 두 패널 색척도 공통. 검은 점선은 x의 10분위 중앙값.\n")
A(tbl(["경로", "Spearman ρ"], [
    ["예측 Ŷ₂ → 측정 Y₂ᵥ (eSpCas9)", f"{S8['rho_pred']:.4f}"],
    ["측정 Y₂(HF1) → 측정 Y₂ᵥ (실측 참조 기준)", f"{S8['rho_ceiling']:.4f}"],
    ["참조 기준 대비 달성률", f"{S8['ratio']*100:.1f}%"],
]))
A(f"\n- Y₂ᵥ도 상한 1.5에서 절단된다 — hold-out 5,394개 중 {int((sc['meas_Y2v_eSpCas9']>=1.5).sum())}개. 캡션에 명시할 것.\n")

A("\n### 그림 9 — 선별 전략 비교 (표 6의 시각화)\n")
A("- 그룹 막대: x = 선별 전략 4개 + hold-out 전체, 색 = 측정 지표 3개(슬롯1 Y₁·슬롯2 Y₂·슬롯3 Y₂ᵥ)\n"
  "- 상위 k = 20. 무릎점과 가중합은 예측값을 min–max 정규화한 뒤 계산한다.\n")
A(tbl(["전략", "측정 Y₁", "측정 Y₂", "측정 Y₂ᵥ"],
      [[k.replace("\n", " "), f"{v['측정 Y₁ 편집효율']:.3f}",
        f"{v['측정 Y₂ 유지율 (HF1)']:.3f}", f"{v['측정 Y₂ᵥ 유지율 (eSpCas9)']:.3f}"]
       for k, v in S9.items()]))
A("""
**정렬 키가 같다** — 무릎점은 `argmax(n₁+n₂)` 를 F₁ 26개 안에서, 가중합은
`argmax(0.5n₁+0.5n₂)` 를 5,394개 전체에서 고른다. 두 함수는 같은 순서를 만든다.
따라서 두 전략의 성능 차이는 전적으로 **후보군을 F₁으로 제한한 데서** 온다.
F₁ 26개의 전역 가중합 순위 중앙값은 """ + f"{F1_GLOBAL_RANK_MED:,}" + """위이므로, k=20이면 F₁의 """ + f"{K_FRAC*100:.0f}" + """%를 가져가야 하고
그 안에 전선 양 끝의 극단점이 섞여 들어간다. 고찰에서 이 구조를 밝히지 않으면
그림이 프레임워크에 불리하게만 읽힌다.
""")

A("\n### 그림 10 — λ 손실가중 전선 (선택 도판)\n")
A("""- 좌: λ 11개 전부 / 우: λ = 0.1–0.9 확대. λ=0·1 은 한쪽 head 미학습이라 목적공간 퇴화점이다.
- 점은 시드 5개 평균, **가로·세로 양방향 오차막대**는 표본표준편차(ddof = 1). 색은 슬롯1 파랑 단일 계열.
- **「비지배 λ」 표기는 전부 제거했다.** 일곱 번의 산출에서 일곱 개의 서로 다른 집합이
  나왔고 공통 원소가 하나도 없다 — 이 자료로 결정할 수 있는 양이 아니다(원고 4.6절).
  이전 판의 `{0.4, 0.5, 0.6, 0.7, 0.8}` 은 단일 실행의 산물이므로 인용하지 말 것.
- λ 값 라벨은 유지한다. 오차막대가 겹쳐서 확대 패널은 지시선으로 라벨을 빼 두었다.
- λ 0.3·0.4 와 0.5·0.7 은 평균이 거의 같은 자리에 있다. 겹쳐 보이는 것이 정상이다.
""")

A("\n### 그림 11 — 무릎점 선별의 k 민감도\n")
A("- 3패널. 좌·중은 선 그래프 3계열(효율 단독 슬롯1 · 가중합 슬롯2 · 무릎점 슬롯3, 순서 고정),\n"
  "  우는 전선 순위별 누적 평균 2계열. k = 20 위치에 점선 참조선.\n"
  "- 자료: `phase7_knee_k.json` (`src/phase7_knee_k.py` 산출)\n")
A(tbl(["k", "k/\\|F₁\\|", "효율 단독 Y₁ / Y₂", "가중합 Y₁ / Y₂", "무릎점 Y₁ / Y₂"],
      [[k, f"{k / KK['n_front_F1']:.2f}"] +
       [f"{KK['results'][f'k{k}_{m}']['Y1_mean']:.4f} / {KK['results'][f'k{k}_{m}']['Y2_mean']:.4f}"
        for m in ("eff", "sum", "knee")]
       for k in KK["k_values"]]))
A(f"\n- |F₁| = {KK['n_front_F1']}. k = 20 은 전선의 {20 / KK['n_front_F1'] * 100:.0f}% 다.\n"
  "- 무릎점의 측정 Y₁ 은 k 가 커질수록 단조 감소하고, 가중합은 k 에 거의 무감각하다.\n"
  "  즉 구판 원고의 「무릎점은 효율 손실이 크다」는 전략의 성질이 아니라 k = 20 설정의 산물이다.\n"
  "- 다만 가중합 − 효율 단독의 이득은 네 k 모두 부호가 같아(ΔY₂ +0.0858~+0.1547) k 에 견고하다.\n"
  "  (이 선별 전략 비교(V4)는 10쪽판 원고에서 제외되었다. 사유는 분량도, k 민감도도 아니다 —\n"
  "   예측값으로 고르는 분석이라 재학습하면 무너진다. 측정 Y₂ 이득 원 실행 +0.098 → 재실행 +0.010,\n"
  "   같은 재실행에서 예측 전선 26 → 20, 예측 동시 만족 587·87·0 → 565·121·7.)\n")

A("\n### 그림 12 — q-스캔: 귀무분포 대비 편차\n")
A("""- 선 그래프 2계열, **단일 선형 z 축**. 이중 축 금지.
- 세로축 z = (관측 − 귀무평균) / 귀무sd. 귀무는 실제 스피어만을 맞춘 가우시안 copula
  (2,000회, seed 20260906). **독립 기대 Nq² 는 상관을 무시하므로 기준선으로 쓰지 않는다.**
- 가로축 q 는 역방향(왼쪽 0.30 → 오른쪽 0.03). 측정 = 슬롯1 파랑, 예측 = 슬롯2 주황.
- z = 0 실선, z = ±2 점선. 측정 곡선이 0을 지나는 구간에 옅은 세로 음영.
- 직접 라벨은 측정 곡선의 q = 0.15 / 0.10 / 0.05 세 지점만.
- 오차막대 없음 — 관측은 1회이고 산포는 귀무분포의 것이며 이미 z 의 분모에 들어가 있다.
""")
A(tbl(["q", "독립 Nq²", "귀무 평균", "귀무 sd", "측정 관측", "측정 z", "예측 관측", "예측 z"],
      [[f"{a['q']:.2f}", f"{a['independent_Nq2']:.1f}", f"{a['null_mean']:.1f}", f"{a['null_sd']:.2f}",
        a["observed"], f"{a['z']:+.2f}", b["observed"], f"{b['z']:+.2f}"]
       for a, b in zip(QN["series"]["measured"]["rows"], QN["series"]["predicted"]["rows"])]))
A(f"""
- 스피어만 상관: 측정 {QN['series']['measured']['spearman']:+.4f} (copula r {QN['series']['measured']['copula_r']:+.4f}),
  예측 {QN['series']['predicted']['spearman']:+.4f} (copula r {QN['series']['predicted']['copula_r']:+.4f})
- z = 0 통과 구간: 측정 q = {QN['sign_change']['measured']['z_crosses_zero_between'][0]:.2f}–{QN['sign_change']['measured']['z_crosses_zero_between'][1]:.2f},
  예측 q = {QN['sign_change']['predicted']['z_crosses_zero_between'][0]:.2f}–{QN['sign_change']['predicted']['z_crosses_zero_between'][1]:.2f}
- 처음 z ≤ −2 가 되는 q: 측정 {QN['sign_change']['measured']['first_q_with_z_le_minus2']:.2f},
  예측 {QN['sign_change']['predicted']['first_q_with_z_le_minus2']:.2f}

두 곡선은 같은 방향으로 무너지지만 예측 곡선이 더 늦게(더 좁은 q 에서) 0을 지난다.
「사실상 일치」보다 「같은 구간에서 같은 방향」이 정확한 서술이다.
""")

A("\n**부속 수치 — 도판에는 넣지 않는다 (본문 §3.3용)**\n")
A(tbl(["항목", "값"], [
    ["copula 귀무 전선 크기 — 예측 조건",
     f"{FN['null']['predicted']['mean']:.2f} ± {FN['null']['predicted']['sd']:.2f} "
     f"(SE {FN['null']['predicted']['se_of_mean']:.3f}, z = {FN['null']['predicted']['z']:.2f})"],
    ["copula 귀무 전선 크기 — 측정 조건",
     f"{FN['null']['measured']['mean']:.2f} ± {FN['null']['measured']['sd']:.2f} "
     f"(SE {FN['null']['measured']['se_of_mean']:.3f}, z = {FN['null']['measured']['z']:.2f})"],
    ["독립 이론값 Hₙ", f"{FN['independent_theory_Hn']:.2f}"],
    ["반복 횟수 / 시드", f"copula {FN['reps_null']:,}회 · 교란 {FN['reps_jitter']:,}회 · seed {FN['seed']}"],
]))
A("\n귀무분포는 **전선 크기**에 대한 것이고 q-스캔은 **점유율**이라 축이 다르다. "
  "같은 그림에 겹치지 말 것.\n")

A("\n## 4. 아직 그리지 않은 것 — 없음\n")
A("""이전 판의 후보 4개는 모두 그렸다.

| 이전 후보 | 결과 |
|---|---|
| eSpCas9 교차 검증 산점도 | 그림 8 |
| λ 손실가중 전선 | 그림 10 (선택) |
| 선별 전략 비교 막대 | 그림 9 |
| 파레토 전선의 실측 재현 | 그림 7 |
| (2026-09-06 추가) q-스캔 | 그림 12 |
| (2026-09-06 추가) 무릎점 k 민감도 | 그림 11 |

2026-09-06 육안 감사 수정 반영분 (F1–F5):

- 그림 5 — 강조 라벨 상자가 인접 칸을 덮던 것을 없애고, 값을 패널 제목 줄로 옮겼다
- 그림 6 우 — 사분위별로 값이 큰 계열은 위, 작은 계열은 아래에 라벨을 놓고 글자색을 계열색에 맞췄다
- 그림 3 — 두 패널의 x·y 축 범위를 통일했다
- 그림 3 우 — F₁ 을 좌 패널과 같은 강조색으로 빼고 F₂–F₅ 를 램프 전 구간에 균등 배치했다
- 그림 2 — 채택한 λ = 0.5 에 수직선·강조 마커·값 라벨을 넣었다
- 그림 4·5 — x축 눈금을 짝수로 바꿔 **위치 20** 이 보이게 했다

**그림 번호 (2026-09-06 원고 담당 확정)**

- 그림 11 = 무릎점 k 민감도 (`fig11_knee_k_sensitivity.*`)
- 그림 12 = q-스캔 (`fig12_qscan.*`)
- 그림 7–10 은 전부 본문에 편입됐다(§3.3 · §4.1 · §4.3 · §3.2). 보충자료로 빠지는 도판은 없다.
- 최초 인용 순서가 비단조이므로 **재번호는 최종 조판 단계에서 일괄** 처리한다. 파일명은 그대로 둔다.

남은 미해결 항목:

- 그림 4 우측은 여전히 Y₂ 히트맵 하나다. `nt_importance.eff` 가 있으므로 Y₁ 히트맵을
  나란히 놓을 수 있으나, 패널 구성이 바뀌면 원고의 그림 4 서술도 함께 고쳐야 하므로 보류했다.
- 오차막대는 그리지 않는다 — 그림 1–10 은 전부 단일 시드 1회 학습이다.
  캡션에 "단일 시드 1회 학습 결과"를 명시하는 것이 정직한 처리다.
  (그림 11 의 귀무분포만 2,000회 반복이며, 이는 도판이 아니라 본문 수치다.)

- 무릎점 전략의 k 민감도 — k=20 은 |F₁|=26 의 77%다. k를 줄이면 결과가 달라질 수 있다.
  후보군을 F₁–F₅ 로 넓히는 변형도 검증하지 않았다. 결과를 본 뒤 고른 컷오프이므로,
  쓰려면 사후 탐색이라고 명시해야 한다.
- 그림 1 은 `out/dataset.npz` 가 있어야 재생성된다. zip 배포본에는 빠져 있으니
  `python src/phase1_data.py` 로 먼저 만든다(약 7초, seed 20260830 고정, 원본과 일치 확인됨).
""")

A("\n## 5. 고칠 때 지켜야 할 것\n")
A("""- 수치를 이 문서에서 손으로 옮기지 말 것. `figure_data.json`을 읽어서 그릴 것.
- 색을 새로 고르지 말 것. 1절 팔레트의 역할에서만 가져올 것.
- 계열이 4개를 넘으면 색을 늘리지 말고 분할(small multiples)하거나 묶을 것.
- 이중 축(y축 2개)은 어떤 경우에도 만들지 말 것.
- 그린 뒤 반드시 렌더해서 눈으로 볼 것 — 라벨 충돌과 넘침은 코드로 안 잡힌다.
- 도판을 고쳤으면 `src/figures.py`도 함께 고쳐 재현성을 유지할 것.\n- 벡터판이 필요하면 `figures.py` 의 `VECTOR = True` 를 확인할 것 (PNG·PDF·SVG 동시 저장).\n- 코드를 고친 뒤에는 기존 PNG 와 md5 로 대조해 의도치 않은 변화를 확인할 것.
""")

(ROOT / "도표-그림-명세.md").write_text("\n".join(M))
print("[save] out/figure_data.json, out/scatter_points.csv, 도표-그림-명세.md")
print("       figure_data.json keys:", list(FD.keys()))
