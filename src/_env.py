"""
공용 환경 해석 — 데이터 경로와 연산 장치
========================================
전 스크립트가 이 모듈을 import 한다. 하드코딩 절대경로와 스크립트별 장치 분기를
한곳으로 모은다. (2026-09-12, 제3환경 재현 실패 3건에 대한 조치)

데이터 경로 우선순위
  1. 환경변수 KSEF_DATA_DIR
  2. <저장소 루트>/../data                                  (맥 폴더 구조)
  3. <저장소 루트>/data                                     (zip 을 그대로 푼 경우)
  4. /mnt/user-data/uploads/Documents/KSEF_CRISPR/data       (구 작업공간, 하위호환)

연산 장치
  **기본값은 cpu 다. 원고 정본 수치가 CPU 산출물이기 때문이다.**
  KSEF_DEVICE=auto 를 주면 cuda → mps → cpu 순으로 고른다.
  장치를 바꾸면 부동소수점 누적 순서가 달라져 학습 결과가 미세하게 달라지므로,
  원고 수치를 재현하려면 기본값 그대로 실행할 것.
"""
import os
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_MARKER = "deephf_3enzyme_derived.csv"


def data_dir(marker=_MARKER):
    """원자료 폴더를 찾아 반환한다. 못 찾으면 후보 목록과 함께 즉시 실패한다."""
    cands = []
    env = os.environ.get("KSEF_DATA_DIR")
    if env:
        cands.append(pathlib.Path(env).expanduser())
    cands += [_ROOT.parent / "data",
              _ROOT / "data",
              pathlib.Path("/mnt/user-data/uploads/Documents/KSEF_CRISPR/data")]
    for c in cands:
        if (c / marker).exists():
            return c
    raise FileNotFoundError(
        f"원자료 폴더를 찾지 못했다. '{marker}' 가 있는 폴더를 "
        f"환경변수 KSEF_DATA_DIR 로 지정하라.\n시도한 경로:\n  " +
        "\n  ".join(str(c) for c in cands))


def device():
    """연산 장치를 반환한다. 기본 'cpu' — 원고 정본 조건."""
    d = os.environ.get("KSEF_DEVICE", "cpu")
    if d != "auto":
        return d
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
