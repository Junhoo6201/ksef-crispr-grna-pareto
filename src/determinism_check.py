"""
결정성 확인 — GPU 실행 간 변동의 원인 규명 (2026-09-13)
=======================================================
사전 등록: claude/사전등록-시드확장-2026-09-12.md 부록 C

질문: 같은 시드로 두 번 학습했을 때 결과가 달라지는가. 달라진다면 GPU 비결정
      커널이 원인인가.

설계 (실행 전 고정)
  조건 A : 기본 설정 그대로, λ=0.5 · 시드 0 을 연속 2회 학습
  조건 B : 결정적 연산을 강제하고 같은 것을 연속 2회 학습

판정 (실행 전 고정)
  A1 ≠ A2 이고 B1 = B2  → 원인은 비결정적 GPU 커널 [확정]
  A1 = A2               → 세션 안에서는 결정적. 4.7절이 관측한 세션 간 차이는
                          다른 원인(드라이버 판본 등)이며 본 실험으로는 규명 못 함
  A1 ≠ A2 이고 B1 ≠ B2  → 결정적 설정으로도 안 잡힘. 원인 미상으로 기록

"같다"의 정의: sp_eff 와 sp_fid 가 부동소수점 비트 단위로 완전히 일치.
반올림해서 비교하지 않는다.

출력: out/determinism_check.json (새 파일. 기존 결과 파일은 건드리지 않는다)
"""
import os
# CUDA 결정적 행렬곱에 필요 — torch import 전에 설정해야 한다
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json, pathlib, sys, time
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import phase2_model as P2

LAM, SEED = 0.5, 0


def one(D, tag):
    t0 = time.time()
    _, r = P2.run(LAM, D, seed=SEED)
    r["tag"] = tag
    r["secs"] = round(time.time() - t0, 1)
    print(f"  {tag}: sp_eff {r['sp_eff']!r}", flush=True)
    print(f"      sp_fid {r['sp_fid']!r}  ep {r['epochs_run']}  {r['secs']}s", flush=True)
    return r


def same(a, b):
    return a["sp_eff"] == b["sp_eff"] and a["sp_fid"] == b["sp_fid"]


if __name__ == "__main__":
    D = dict(np.load(P2.ROOT / "out/dataset.npz", allow_pickle=True))
    out = {"device": P2.DEV, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "lam": LAM, "seed": SEED}
    print(f"[env] device = {P2.DEV} | torch {torch.__version__} | {out['gpu']}", flush=True)

    print("\n=== 조건 A : 기본 설정, 연속 2회 ===", flush=True)
    a1, a2 = one(D, "A1"), one(D, "A2")

    print("\n=== 조건 B : 결정적 연산 강제, 연속 2회 ===", flush=True)
    mode = "use_deterministic_algorithms(True)"
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:                       # 일부 연산에 결정적 구현이 없을 때
        mode = f"cudnn.deterministic only ({type(e).__name__}: {e})"
        print(f"  [주의] {mode}", flush=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out["mode_B"] = mode
    try:
        b1, b2 = one(D, "B1"), one(D, "B2")
        b_ok = same(b1, b2)
    except Exception as e:                       # 결정적 커널이 없어 실행 자체가 실패
        b1 = b2 = None
        b_ok = None
        out["B_error"] = f"{type(e).__name__}: {e}"
        print(f"  [실패] {out['B_error']}", flush=True)

    a_ok = same(a1, a2)
    out["runs"] = [r for r in (a1, a2, b1, b2) if r is not None]
    out["A_identical"], out["B_identical"] = a_ok, b_ok

    if not a_ok and b_ok is True:
        v = "원인은 비결정적 GPU 커널 [확정]"
    elif a_ok:
        v = "세션 안에서는 결정적. 세션 간 차이의 원인은 본 실험으로 규명 못 함"
    elif b_ok is False:
        v = "결정적 설정으로도 재현되지 않음. 원인 미상"
    else:
        v = "조건 B 실행 실패 — 판정 불가"
    out["verdict"] = v

    print("\n===== 판정 =====", flush=True)
    print(f"  조건 A 두 실행 동일? {a_ok}", flush=True)
    print(f"  조건 B 두 실행 동일? {b_ok}   (모드: {mode})", flush=True)
    if not a_ok:
        print(f"  A 차이: Δsp_eff {abs(a1['sp_eff']-a2['sp_eff']):.2e}  "
              f"Δsp_fid {abs(a1['sp_fid']-a2['sp_fid']):.2e}", flush=True)
    print(f"  → {v}", flush=True)
    print(f"\n  (참고) 4.7절이 관측한 세션 간 차이: Δsp_eff 4.2e-03  Δsp_fid 3.3e-03", flush=True)

    json.dump(out, open(P2.ROOT / "out/determinism_check.json", "w"), indent=2, ensure_ascii=False)
    print("[save] out/determinism_check.json", flush=True)
