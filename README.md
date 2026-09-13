# CRISPR-Cas9 gRNA의 절단 효율과 고충실도 효소에서의 활성 유지율은 상충하는가

멀티태스크 심층학습과 파레토 분석에 의한 정량적 검증 — KSEF 출품 연구의 코드·데이터·결과 저장소

**공준후 · 고지성 · 정호영**

> Do cleavage efficiency and activity retention in high-fidelity Cas9 conflict at the guide level?
> Code, data and results for a Korean high-school science fair (KSEF) study.
> All analyses are **in silico**; no new wet-lab experiments were performed.

---

## 1. 이 연구가 답하는 질문

gRNA 하나에 대해 두 값을 함께 본다.

| 기호 | 정의 | 평균 (n = 53,937) |
|---|---|---|
| Y₁ | 야생형 SpCas9의 편집 효율 (실측) | 0.7248 |
| Y₂ | SpCas9-HF1 효율 ÷ 야생형 효율, 상한 1.5로 절단 (실측) | 0.6666 |
| Y₂ᵥ | eSpCas9(1.1) 효율 ÷ 야생형 효율 — 학습 미사용, 외부 검증용 | 0.4842 |

**질문**: 가이드 수준에서 Y₁과 Y₂는 상충하는가. 상충한다면 어느 구간부터인가.

**핵심 관찰**: 두 지표는 전역적으로 거의 무상관이며(평가셋 Spearman ρ = +0.0324),
상충은 상위 10% 부근 아래에서만 나타난다. 관측 상관을 맞춘 가우시안 copula 귀무분포 대비
상위 25%에서는 z = +4.78로 오히려 과잉, 상위 5%에서는 z = −3.41로 결핍이다.

**이 판정은 모델 예측이 아니라 실측값만으로 계산된다.** 그래서 학습을 다시 해도 바뀌지 않는다.

---

## 2. 폴더 구조

```
src/    전처리·학습·해석·검증·도판 생성 코드 (17개)
data/   Phase 1 의 입력 원자료 2개
out/    모든 결과 파일 · 학습된 대표 모델 · 사전 등록 문서
fig/    그림 12장 (png · pdf · svg)
```

---

## 3. 실행 순서

Python 3.13 · PyTorch 2.11 · NumPy 2.1 · SciPy 1.16 · pandas 2.2 · matplotlib 에서 확인하였다.

```bash
python src/phase1_data.py       # 전처리와 누출 없는 분할 → out/dataset.npz
python src/phase2_model.py      # λ 스윕 11개 모델 → out/phase2_lambda_sweep.json, model_lam050.pt
python src/phase3_pareto.py     # 비지배 정렬과 파레토 전선
python src/phase3_front_null.py # 전선 크기의 귀무분포
python src/phase3_qscan_null.py # 분위별 동시 만족 가이드의 귀무분포
python src/phase4_interpret.py  # saliency 와 가상 돌연변이
python src/phase5_validate.py   # 독립 검증 V1·V2·V4
python src/phase6_baseline.py   # 사전 등록 비교 V5
python src/phase7_knee_k.py     # 선별 개수 k 민감도
python src/figures.py           # 그림 12장
```

시드 반복과 결정성 대조는 별도로 실행한다.

```bash
python src/phase2_seed_sweep.py   # λ 11개 × 시드 5개 = 55 모델
python src/phase6_seed_sweep.py   # 시드 베이스 5개 × 8 모델 = 40 모델
python src/phase2_seed_agg.py     # 위 결과를 λ별 평균 ± 표본표준편차로 집계
python src/phase6_seed_agg.py
python src/determinism_check.py   # 같은 시드 2회 반복 × 기본/결정적 설정
```

**`out/dataset.npz` 는 배포본에 포함되어 있지 않다.** `phase1_data.py` 를 먼저 실행해야
나머지 단계가 동작한다.

### 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `KSEF_DEVICE` | `cpu` | `auto` 를 주면 cuda → mps → cpu 순으로 고른다. **본문 수치는 기본값에서 산출되었다.** |
| `KSEF_DATA_DIR` | 자동 탐색 | `data/` 를 다른 곳에 두었을 때 지정한다. |

### 난수 시드

Phase 1–5는 0, Phase 6은 4(모델별 1·2·3·4), 귀무분포와 교란은 20260906,
시드 반복은 Phase 2가 0–4 · Phase 6이 시드 베이스 0·10·20·30·40이다.
`phase7_knee_k.py` 는 난수를 쓰지 않는다(18행에 선언된 시드는 사용되지 않는다).

---

## 4. 재현의 층위 — 먼저 읽을 것

이 저장소를 받아 실행하면 **어떤 수치는 정확히 재현되고 어떤 수치는 재현되지 않는다.**
셋으로 나뉜다.

**① 측정값만으로 계산되는 양 — 항상 같다.**
측정 기준 전선 크기(11개), 분위별 동시 만족 가이드 수(423 / 55 / 3), 평가셋 상관,
전략별 실측 성능의 기준선. 다른 기계·다른 환경에서 전 과정을 다시 돌려도 한 자리도 변하지 않았다.
**논문의 핵심 관찰이 전부 여기에 속한다.**

**② 저장된 모델 파일을 쓰면 예측값도 재현된다.**
`out/model_lam050.pt` 를 그대로 쓰고 Phase 3 이후를 실행하면 예측 기준 수치가 재현된다.
가중치를 NumPy로 독립 재구현해 순전파한 결과가 최대차 2.4×10⁻⁷로 일치했고,
다른 기계에서 파이프라인을 다시 돌려도 같은 값이 나왔다.

**③ 학습부터 다시 하면 예측값 기반 수치는 재현되지 않는다.**
예측 전선 크기, 선별 이득, 가상 돌연변이 상위 치환 목록 등이 달라진다.
원인은 시드가 아니라 **GPU의 비결정적 연산**이다 — 같은 시드로 두 번 학습했을 때
결과가 달랐고(ΔY₁ 2.0×10⁻³), `torch.use_deterministic_algorithms(True)` 를 켜자
두 실행이 부동소수점 비트 단위로 일치했다(`out/determinism_check.json`).
같은 실행 안에서 시드를 바꾼 표본표준편차는 0.0009로, 실행 간 차이보다 작다.

**요약: 논문의 결론을 확인하려면 ①을 보면 되고, 본문의 예측값 수치를 그대로 얻으려면
학습을 다시 하지 말고 `out/model_lam050.pt` 를 쓰면 된다.**

---

## 5. 사전 등록

결과를 보기 전에 가설과 반증 조건을 문서로 고정한 절차를 두 번 사용하였다.

- `out/preregistration.md` — 사전 등록 비교 P1·P2·P3 (2026-08-30)
- 시드 확장과 결정성 대조의 사전 등록은 논문 부록에 수록되어 있다.

**판정 결과를 그대로 공개한다.** P1은 지지되었고, P2는 판정 불가, P3은 기각되었다.
기각된 예측을 근거로 쓰지 않았다.

---

## 6. 데이터 출처와 라이선스

**원자료**는 다음 논문의 보충자료에서 취득하여 파생한 것이다.

> Wang, D., Zhang, C., Wang, B. *et al.* (2019). Optimized CRISPR guide RNA design for two
> high-fidelity Cas9 variants by deep learning. *Nature Communications* **10**, 4284.
> https://doi.org/10.1038/s41467-019-12281-8 — **CC BY 4.0**

`data/` 의 두 CSV 는 위 보충자료(Supplementary Data 1·2)에서 파생한 것이며
원저작자 표시 조건 아래 **CC BY 4.0** 으로 재배포한다.
세 효소 공통 20-mer 53,937개가 분석 대상이다.
`out/` 의 결과 파일도 이 데이터에서 파생하였으므로 같은 조건을 따른다.

**코드**(`src/`)는 MIT 라이선스다. `LICENSE` 참고.

새로 생성한 측정 데이터는 없다.

---

## 7. 알려진 한계

논문 본문에 전부 적었다. 요약하면 다음과 같다.

- Y₂는 오프타겟을 직접 잰 값이 아니라 고충실도 효소에서의 **활성 유지율**이다.
  특이성은 가이드가 아니라 효소가 제공한다.
- 세 데이터셋 모두 HEK293T 세포에 삽입한 합성 표적 라이브러리에서 나왔다.
- DeepHF 는 플라스미드 기반 측정이므로 리보핵단백질(RNP) 전달 조건으로 일반화되지 않는다.
- 본문 수치를 낸 실행은 시드 하나짜리이며, 그 실행의 라이브러리 버전 기록이 남아 있지 않다.
  이 공백을 보완하기 위해 사양이 완전히 기록된 환경에서 전 과정을 다시 수행하였다
  (`out/env_seed_sweep.json`).
- 비율 지표 Y₂ = HF1/WT 는 분모가 작을 때 불안정하다. 상한 1.5로 절단하였다(53,937건 중 161건).

---

## 8. 생성형 AI 사용 기록

> **업로드 전에 사실 여부를 직접 확인하고 필요하면 고칠 것.** 아래는 초안이다.

이 연구에서 Anthropic 의 Claude 를 다음 범위로 사용하였다.

- 한국어 원고의 초안 작성과 문장 다듬기
- 본문 수치와 결과 파일의 대조, 계산 재실행에 의한 검증
- 코드 검토와 결함 진단, 분석 스크립트 작성
- 참고문헌 서지정보의 PubMed 대조

연구 질문의 설정, 데이터셋 선택, 사전 등록 내용의 결정, 결과의 해석과 최종 판단은
연구자가 수행하였다. AI 가 제시한 수치는 원자료나 PubMed 에서 다시 확인한 것만 사용하였으며,
확인되지 않은 것은 폐기하였다. 실제로 AI 가 기억에 의존해 제시한 참고문헌 식별자 가운데
세 건이 다른 논문으로 밝혀져 폐기하였고, 이 사례를 검증 절차의 근거로 삼았다.

---

## 9. 인용

이 저장소를 인용할 때는 논문과 원자료 논문(Wang et al., 2019)을 함께 밝혀 주기 바란다.
