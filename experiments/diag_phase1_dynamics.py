"""identity-prior 모델 학습 실패 원인 확정 — [A] single-image overfit + [B] e9 동역학.

읽기/추론 + **작은 overfit only** (본학습 X, 체크포인트 저장 X, 코드 수정 X).
모든 학습은 메모리상 임시 모델에서만 수행하며 파일을 덮어쓰지 않는다.

[A] 결정 실험: LOL 학습쌍 1개에 fresh identity-prior 모델을 800 step 과적합.
    PSNR 25~35dB 도달 → 아키텍처 학습가능. 7~12dB 정체 → affine head grad 단절.

[B] e9 (runs/bilateral_phase1_idprior_l1only/checkpoints/last.pth) 재진단:
    1. to_grid raw 예측(slice 전) scale/bias 통계 (init≈0 에서 움직였나).
    2. to_grid 가중치 L2: 실제 init(seed42 재현) vs e9, delta L2.
    3. 10 step L1-only 중 모듈별 grad norm + param-delta (grad 있는데 안 움직이나).
    4. train_log.csv 의 epoch 1/3/5/7/9 val_psnr.

경로: configs/paths.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import copy
import csv as csvmod

import torch
import torch.nn.functional as F

from src.data.lowlight_dataset import build_dataset_by_name
from src.models.bilateral_grid import build_from_config, BilateralLowLightNet
from src.utils.metrics import psnr_metric
from src.utils.paths import load_paths

HR = "=" * 78
SUB = "-" * 78

# train.py 가 사용한 모델/시드 (config 와 동일)
MODEL_CFG = {"model": {"cw": 32, "grid_size": 16, "depth": 8, "low_res": 256,
                       "refine_channels": 24, "refine_blocks": 2,
                       "guidance_channels": 16, "norm": "in"}}
SEED = 42


def _l2(params) -> float:
    s = 0.0
    for p in params:
        s += float(p.detach().float().pow(2).sum())
    return s ** 0.5


def _grad_norm(params) -> float:
    s = 0.0
    for p in params:
        if p.grad is not None:
            s += float(p.grad.detach().float().pow(2).sum())
    return s ** 0.5


def _param_delta(before: dict, module, prefix: str) -> float:
    s = 0.0
    for n, p in module.named_parameters():
        key = prefix + n
        s += float((p.detach() - before[key]).float().pow(2).sum())
    return s ** 0.5


def _snapshot(module, prefix: str) -> dict:
    return {prefix + n: p.detach().clone() for n, p in module.named_parameters()}


# ===========================================================================
# [A] single-image overfit
# ===========================================================================
def run_overfit(device: str) -> None:
    print(HR)
    print(" [A] single-image overfit (fresh identity-prior 모델, 800 step, fp32)")
    print(HR)
    P = load_paths()
    # LOL v1 train(our485)에서 고정 1쌍 (augment off → 256 resize, 결정론적)
    ds = build_dataset_by_name("lol_v1", data_root=P.lol_v1, split="train",
                               image_size=256, augment=False, full_resize=False)
    low, high = ds[0]
    low = low.unsqueeze(0).to(device)
    high = high.unsqueeze(0).to(device)
    print(f"  학습쌍: lol_v1/train idx0  shape={tuple(low.shape)}")
    print(f"  baseline PSNR(low vs high) = {psnr_metric(low.clamp(-1,1), high):.3f} dB")
    print(SUB)

    torch.manual_seed(SEED)
    model = BilateralLowLightNet(**MODEL_CFG["model"]).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # AMP off, fp32

    print(f"  {'step':>5} | {'L1':>8} | {'PSNR(out vs high)':>18}")
    print(SUB)
    for step in range(1, 801):
        opt.zero_grad(set_to_none=True)
        out = model(low)
        loss = F.l1_loss(out, high)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                ps = psnr_metric(out.clamp(-1, 1), high)
            print(f"  {step:>5} | {loss.item():>8.5f} | {ps:>18.3f}")

    with torch.no_grad():
        final_ps = psnr_metric(model(low).clamp(-1, 1), high)
    print(SUB)
    verdict = ("아키텍처 학습가능 (overfit 성공)" if final_ps >= 25
               else "affine head/학습경로 문제 (overfit 실패)" if final_ps <= 13
               else "부분적 — 추가 분석 필요")
    print(f"  최종 PSNR = {final_ps:.3f} dB  →  판정: {verdict}")
    print(HR)


# ===========================================================================
# [B] e9 재진단
# ===========================================================================
def run_e9_diag(device: str) -> None:
    P = load_paths()
    ckpt = P.runs / "bilateral_phase1_idprior_l1only" / "checkpoints" / "last.pth"
    print(HR)
    print(" [B] e9 체크포인트 + 학습 동역학 재진단")
    print(HR)
    print(f"  checkpoint : {ckpt}")
    if not ckpt.is_file():
        print(f"  [error] 없음: {ckpt}")
        return
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = state.get("config") or MODEL_CFG
    model = build_from_config(cfg).to(device).eval()
    model.load_state_dict(state["model"], strict=False)
    print(f"  epoch={state.get('epoch')}  global_step={state.get('global_step')}  "
          f"best_psnr={state.get('best_psnr')}")
    print(SUB)

    # eval15 샘플 5장
    ds_eval = build_dataset_by_name("lol_v1", data_root=P.lol_v1, split="eval",
                                    image_size=256, augment=False)
    xe = torch.stack([ds_eval[i][0] for i in range(5)]).to(device)

    # --- [B].1 to_grid raw 예측 (slice 전 grid) scale/bias 통계 ---
    print(" [B].1 to_grid raw 예측 (slice 전) — identity prior 는 apply_affine 에서 +1,")
    print("        따라서 raw grid 의 scale/bias 가 0 에서 움직였는지 본다.")
    with torch.no_grad():
        x_low = model._downsample_for_coeff(xe)
        grid = model.coefficient_net(x_low)        # (N,12,depth,gh,gw)
    N, _, D, gh, gw = grid.shape
    G = grid.view(N, 3, 4, D, gh, gw)
    scale_block = G[:, :, :3]                       # 9 scale 성분 (대각 포함)
    diag = torch.stack([G[:, 0, 0], G[:, 1, 1], G[:, 2, 2]])
    bias_block = G[:, :, 3]
    def st(t):
        t=t.float(); return f"mean={t.mean():+.5f} std={t.std():.5f} min={t.min():+.4f} max={t.max():+.4f}"
    print(SUB)
    print(f"   scale 전체(9성분) : {st(scale_block)}")
    print(f"   scale 대각(3성분) : {st(diag)}   (+1 후 유효 scale ≈ {diag.float().mean()+1:.4f})")
    print(f"   bias(3성분)       : {st(bias_block)}")
    moved = scale_block.abs().mean().item() > 1e-3 or bias_block.abs().mean().item() > 1e-3
    print(f"   → raw 예측이 0 에서 {'움직임' if moved else '거의 안 움직임 (≈0 정체)'}")
    print(SUB)

    # --- [B].2 to_grid 가중치 L2: 실제 init(seed42) vs e9 ---
    print(" [B].2 to_grid head 가중치: init(seed42 재현) vs e9")
    torch.manual_seed(SEED)
    model_init = build_from_config(cfg).to(device)   # train.py 와 동일 시드 → 동일 init
    init_params = list(model_init.coefficient_net.to_grid.parameters())
    e9_params = list(model.coefficient_net.to_grid.parameters())
    l2_init = _l2(init_params)
    l2_e9 = _l2(e9_params)
    delta = (sum(float((a.detach() - b.detach()).float().pow(2).sum())
                 for a, b in zip(e9_params, init_params))) ** 0.5
    print(SUB)
    print(f"   ||to_grid_init||  = {l2_init:.6f}")
    print(f"   ||to_grid_e9||    = {l2_e9:.6f}")
    print(f"   ||e9 - init||(Δ)  = {delta:.6f}   "
          f"(상대변화 {delta/(l2_init+1e-9)*100:.2f}%)")
    print(SUB)

    # --- [B].3 10 step L1-only 동역학: grad norm + param-delta ---
    print(" [B].3 10 step L1-only 동역학 (e9 메모리 복사본, 저장 안 함, lr=2e-4)")
    dyn = copy.deepcopy(model).train()
    opt = torch.optim.Adam(dyn.parameters(), lr=2e-4, betas=(0.9, 0.999))
    ds_tr = build_dataset_by_name("lol_v1", data_root=P.lol_v1, split="train",
                                  image_size=256, augment=True)
    loader = torch.utils.data.DataLoader(ds_tr, batch_size=8, shuffle=True,
                                         num_workers=0, drop_last=True)
    it = iter(loader)
    mods = {
        "CoeffNet": dyn.coefficient_net,
        "to_grid ": dyn.coefficient_net.to_grid,
        "Guidance": dyn.guidance_net,
        "refine  ": torch.nn.ModuleList([dyn.refine, dyn.refine_out]),
    }
    print(SUB)
    hdr = f"  {'step':>4} | {'L1':>7} |" + "".join(
        f" {m}:g/Δp |" for m in mods)
    print(hdr)
    for step in range(1, 11):
        try:
            low, high = next(it)
        except StopIteration:
            it = iter(loader); low, high = next(it)
        low, high = low.to(device), high.to(device)
        snaps = {name: _snapshot(mod, name) for name, mod in mods.items()}
        opt.zero_grad(set_to_none=True)
        out = dyn(low)
        loss = F.l1_loss(out, high)
        loss.backward()
        gnorms = {name: _grad_norm(mod.parameters()) for name, mod in mods.items()}
        opt.step()
        deltas = {name: _param_delta(snaps[name], mod, name)
                  for name, mod in mods.items()}
        cells = "".join(f" {gnorms[m]:.1e}/{deltas[m]:.1e} |" for m in mods)
        print(f"  {step:>4} | {loss.item():>7.4f} |{cells}")
    print("   (g=grad norm, Δp=param L2 변화량. grad>0 인데 Δp≈0 이면 그 모듈 정체)")
    print(SUB)

    # --- [B].4 train_log.csv val_psnr (epoch 1/3/5/7/9) ---
    print(" [B].4 train_log.csv val_psnr @ epoch 1/3/5/7/9")
    csv_path = P.runs / "bilateral_phase1_idprior_l1only" / "logs" / "train_log.csv"
    print(SUB)
    if csv_path.is_file():
        rows = list(csvmod.DictReader(open(csv_path, encoding="utf-8")))
        for e in (1, 3, 5, 7, 9):
            r = next((x for x in rows if int(x["epoch"]) == e), None)
            if r:
                print(f"   epoch {e}: val_psnr={r['val_psnr']:>10}  val_ssim={r['val_ssim']:>10}  "
                      f"train_l1={r['train_l1']}")
        vals = [r["val_psnr"] for r in rows if r["val_psnr"] not in ("nan", "")]
        uniq = set(round(float(v), 4) for v in vals)
        print(f"   → 기록된 val_psnr 고유값: {sorted(uniq)}  "
              f"({'전부 동일=출력 불변' if len(uniq)==1 else '변화 있음'})")
    else:
        print(f"   [warn] 없음: {csv_path}")
    print(HR)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(HR)
    print(" identity-prior 학습 실패 원인 확정 진단 (read + 작은 overfit only)")
    print(f"  device: {device}")
    run_overfit(device)
    run_e9_diag(device)
    print(" 진단 종료 — 체크포인트/코드 변경 없음.")
    print(HR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
