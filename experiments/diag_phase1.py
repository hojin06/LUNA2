"""Phase 1 학습 실패 정적 진단 — 읽기/추론 전용 (학습·수정 절대 금지).

목적
----
``runs/bilateral_phase1_sanity/checkpoints/last.pth`` (epoch29) 을 로드하여
BilateralLowLightNet 이 왜 향상에 실패하는지(passthrough no-op / 입력 무시 /
affine 고정 / grad 미흐름 / 좌표 컨벤션 오류 등)를 정적으로 점검한다.

데이터: LOL eval15 고정 샘플 5장 (dataset eval-path, resize 256, 결정론적).
경로  : configs/paths.yaml.

이 스크립트는 모델 가중치를 수정하지 않으며 optimizer.step 도 하지 않는다.
[6] 의 backward 는 grad norm 관찰용이며 step 을 호출하지 않으므로 가중치 불변.
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

import torch
import torch.nn.functional as F

from src.data.lowlight_dataset import build_dataset_by_name
from src.models import bilateral_grid as BG
from src.models.bilateral_grid import (
    BilateralLowLightNet, apply_affine, build_from_config, slice_grid,
)
from src.utils.paths import load_paths

HR = "=" * 78
SUB = "-" * 78


def _stats(t: torch.Tensor) -> str:
    t = t.detach().float()
    return (f"min={t.min().item():+.4f}  max={t.max().item():+.4f}  "
            f"mean={t.mean().item():+.4f}  std={t.std().item():.4f}")


def flag(cond_ok: bool, msg_ok: str = "OK", msg_bad: str = "WARN") -> str:
    return f"[{'OK ' if cond_ok else 'WARN'}] {msg_ok if cond_ok else msg_bad}"


def main() -> int:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P = load_paths()
    ckpt_path = P.runs / "bilateral_phase1_sanity" / "checkpoints" / "last.pth"

    print(HR)
    print(" Phase 1 정적 진단 (읽기/추론 전용) — BilateralLowLightNet")
    print(HR)
    print(f"  checkpoint : {ckpt_path}")
    print(f"  device     : {device}")
    if not ckpt_path.is_file():
        print(f"\n[error] 체크포인트 없음: {ckpt_path}")
        return 1

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("config") or {"model": state.get("model_cfg", {})}
    model = build_from_config(cfg).to(device).eval()
    missing = model.load_state_dict(state["model"], strict=False)
    print(f"  epoch      : {state.get('epoch')}   global_step: {state.get('global_step')}")
    print(f"  best_psnr  : {state.get('best_psnr')}   best_ssim: {state.get('best_ssim')}")
    if missing.missing_keys or missing.unexpected_keys:
        print(f"  [warn] state_dict mismatch missing={missing.missing_keys} "
              f"unexpected={missing.unexpected_keys}")
    print(f"  model_cfg  : {cfg.get('model')}")

    # --- 고정 eval15 샘플 5장 ---
    ds = build_dataset_by_name("lol_v1", data_root=P.lol_v1, split="eval",
                               image_size=256, augment=False, full_resize=False)
    idxs = list(range(min(5, len(ds))))
    lows, highs = [], []
    for i in idxs:
        lo, hi = ds[i]
        lows.append(lo); highs.append(hi)
    x = torch.stack(lows).to(device)     # (N,3,256,256) in [-1,1]
    target = torch.stack(highs).to(device)
    print(f"  eval15 샘플: {idxs}  (n={x.size(0)}, {tuple(x.shape[1:])})")
    print(HR)

    # =====================================================================
    # [0] 데이터 컨벤션 + metrics.evaluate clamp/변환
    # =====================================================================
    print("[0] 데이터 컨벤션 / metrics 변환")
    print(SUB)
    print(f"  dataloader input  : {_stats(x)}")
    print(f"  dataloader target : {_stats(target)}")
    in_is_pm1 = x.min().item() < -0.2 and x.max().item() > 0.2
    print(f"  → 범위 판정: dataloader 출력은 {'[-1,1]' if in_is_pm1 else '[0,1]? (확인필요)'}")
    print("  metrics.evaluate() 경로 (src/utils/metrics.py):")
    print("    · fake = generator(low).clamp(-1.0, 1.0)      # 출력 [-1,1] 로 clamp")
    print("    · psnr_metric/ssim_metric: _to_01(x)=((x+1)*0.5).clamp(0,1)  # [-1,1]→[0,1]")
    print("    · data_range=1.0 로 PSNR/SSIM 계산")
    print(HR)

    # =====================================================================
    # [1] 출력 통계
    # =====================================================================
    with torch.no_grad():
        out = model(x)
    print("[1] 출력/입력/타깃 통계")
    print(SUB)
    print(f"  input  : {_stats(x)}")
    print(f"  output : {_stats(out)}")
    print(f"  target : {_stats(target)}")
    over = ((out < -1) | (out > 1)).float().mean().item() * 100
    print(f"  output 가 [-1,1] 벗어난 픽셀 비율: {over:.2f}%")
    print(HR)

    # =====================================================================
    # [2] 출력 vs 입력 (passthrough 여부)
    # =====================================================================
    d_out_in = (out - x).abs().mean().item()
    print("[2] 출력 vs 입력 — mean|out - input|")
    print(SUB)
    print(f"  mean|out-input| = {d_out_in:.6f}")
    print(f"  {flag(d_out_in > 0.02, '입력과 의미있게 다름', 'passthrough no-op 의심 (≈0)')}")
    print(HR)

    # =====================================================================
    # [3] 입력 의존성 (출력 레벨)
    # =====================================================================
    x1, x2 = x[0:1], x[1:2]
    with torch.no_grad():
        o1, o2 = model(x1), model(x2)
    diff_in = (x1 - x2).abs().mean().item()
    diff_out = (o1 - o2).abs().mean().item()
    ratio = diff_out / (diff_in + 1e-9)
    print("[3] 입력 의존성 (출력)")
    print(SUB)
    print(f"  diff_in =mean|x1-x2|   = {diff_in:.6f}")
    print(f"  diff_out=mean|o1-o2|   = {diff_out:.6f}   (ratio out/in = {ratio:.3f})")
    print(f"  {flag(ratio > 0.3, '출력이 입력에 반응', 'diff_out≪diff_in → 입력 무시 의심')}")
    print(HR)

    # =====================================================================
    # [4] affine 계수맵 통계 (slice 후)
    # =====================================================================
    with torch.no_grad():
        inter = model.intermediate(x)
    coeffs = inter["coeffs"]                       # (N,12,H,W)
    N, _, H, W = coeffs.shape
    A = coeffs.view(N, 3, 4, H, W)
    print("[4] affine 계수맵 통계 (slice 후, coeffs→(N,3,4,H,W))")
    print(SUB)
    for c in range(3):
        sc = A[:, c, c]
        print(f"  scale[{c}{c}] : {_stats(sc)}")
    for c in range(3):
        bi = A[:, c, 3]
        print(f"  bias[{c}]   : {_stats(bi)}")
    scdiag = torch.stack([A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]])
    biasall = A[:, :, 3]
    sc_ok = abs(scdiag.mean().item() - 1.0) < 0.5
    bi_ok = abs(biasall.mean().item()) < 0.5
    print(f"  {flag(sc_ok and bi_ok, 'scale≈1 / bias≈0 근처 (정상 범위)', 'scale/bias 비정상 (학습 붕괴 의심)')}")
    print(HR)

    # =====================================================================
    # [5] identity-affine 강제 테스트
    # =====================================================================
    print("[5] identity-affine 강제 (refine zero-init 이면 5a·5b 둘 다 input≈)")
    print(SUB)
    ident = torch.zeros(N, 12, H, W, device=device)
    iv = ident.view(N, 3, 4, H, W)
    iv[:, 0, 0] = 1.0; iv[:, 1, 1] = 1.0; iv[:, 2, 2] = 1.0  # scale=I, bias=0
    with torch.no_grad():
        aff_only = apply_affine(x, ident)                       # (5a)
        r = model.refine(torch.cat([x, aff_only], dim=1))
        residual = model.refine_out(r)
        aff_refine = aff_only + residual                        # (5b)
    d5a = (aff_only - x).abs().mean().item()
    d5b = (aff_refine - x).abs().mean().item()
    res_stat = residual.abs().mean().item()
    print(f"  (5a) apply_affine(identity) vs input        : mean abs diff = {d5a:.6e}")
    print(f"       {flag(d5a < 1e-4, 'identity affine = 정확히 passthrough', 'apply_affine 자체 이상')}")
    print(f"  (5b) +refine vs input                       : mean abs diff = {d5b:.6e}")
    print(f"       refine residual mean|·| = {res_stat:.6e}")
    print(f"       {flag(d5b < 1e-3, 'refine 이 거의 identity (zero-init 유지)', 'refine 이 입력을 크게 변형 (학습으로 이동)')}")
    print(HR)

    # =====================================================================
    # [6] 모듈별 grad norm (L1-only, batch1, step 없음)
    # =====================================================================
    print("[6] 모듈별 grad norm — L1-only 1회 forward+backward (step 안 함)")
    print(SUB)
    model.zero_grad(set_to_none=True)
    xb, tb = x[0:1], target[0:1]
    out_b = model(xb)
    loss = F.l1_loss(out_b, tb)
    loss.backward()

    def gnorm(params) -> float:
        s = 0.0
        for p in params:
            if p.grad is not None:
                s += float(p.grad.detach().pow(2).sum())
        return s ** 0.5

    g_coeff = gnorm(model.coefficient_net.parameters())
    g_guide = gnorm(model.guidance_net.parameters())
    g_refine = gnorm(list(model.refine.parameters()) + list(model.refine_out.parameters()))
    g_head = gnorm(model.coefficient_net.to_grid.parameters())   # affine(grid) head
    print(f"  L1 loss          = {loss.item():.6f}")
    print(f"  CoefficientNet   grad-norm = {g_coeff:.4e}")
    print(f"  GuidanceNet      grad-norm = {g_guide:.4e}")
    print(f"  refine(+out)     grad-norm = {g_refine:.4e}")
    print(f"  affine head(to_grid) grad  = {g_head:.4e}")
    print(f"  {flag(g_head > 1e-8, 'affine head 에 grad 흐름', 'affine head grad≈0 → grid 가 학습 안 됨!')}")
    model.zero_grad(set_to_none=True)
    print(HR)

    # =====================================================================
    # [7] CoefficientNet grid input-dependency
    # =====================================================================
    with torch.no_grad():
        g1 = model.coefficient_net(model._downsample_for_coeff(x1))
        g2 = model.coefficient_net(model._downsample_for_coeff(x2))
    gdiff = (g1 - g2).abs().mean().item()
    gscale = g1.abs().mean().item()
    print("[7] CoefficientNet grid 입력 의존성")
    print(SUB)
    print(f"  mean|grid1-grid2| = {gdiff:.6e}   (grid abs mean = {gscale:.6e})")
    rel = gdiff / (gscale + 1e-9)
    print(f"  relative = {rel:.4f}")
    print(f"  {flag(rel > 0.05, 'grid 가 입력 따라 변함', 'grid 거의 동일 → CoefficientNet 입력 무시')}")
    print(HR)

    # =====================================================================
    # [8] guidance map 통계
    # =====================================================================
    with torch.no_grad():
        gmap = model.guidance_net(x)
    print("[8] guidance map 통계 (GuidanceNet 출력)")
    print(SUB)
    print(f"  guidance : {_stats(gmap)}")
    gm_std = gmap.std().item()
    gm_mean = gmap.mean().item()
    near_const = gm_std < 0.02
    saturated = ((gmap < 0.02) | (gmap > 0.98)).float().mean().item() > 0.9
    print(f"  {flag(not (near_const or saturated), 'guidance 가 다양한 값 분포', '0.5 고정(std≈0) 또는 0/1 포화 → slice 사실상 고정')}")
    if near_const:
        print(f"       → std={gm_std:.4f} (≈0): 거의 상수 {gm_mean:.3f}")
    if saturated:
        print(f"       → 90%+ 픽셀이 0/1 포화")
    print(HR)

    # =====================================================================
    # [9] optimizer / requires_grad 감사
    # =====================================================================
    print("[9] optimizer / requires_grad 감사 (train.py 설정 재현, step 안 함)")
    print(SUB)
    tr = (cfg.get("training") or {})
    lr = float(tr.get("lr", 2e-4)); betas = tuple(tr.get("betas", [0.9, 0.999]))
    wd = float(tr.get("weight_decay", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    opt_ids = {id(p) for grp in opt.param_groups for p in grp["params"]}
    total_params = sum(1 for _ in model.parameters())
    print(f"  optimizer Adam(lr={lr}, betas={betas}, wd={wd})  "
          f"param 텐서 {len(opt_ids)}/{total_params} 포함")

    def in_opt(module) -> str:
        ps = list(module.parameters())
        inc = sum(1 for p in ps if id(p) in opt_ids)
        return f"{inc}/{len(ps)}"

    print(f"  CoefficientNet.to_grid (affine head) : {in_opt(model.coefficient_net.to_grid)} 텐서가 optimizer 에 포함")
    print(f"  GuidanceNet                          : {in_opt(model.guidance_net)}")
    print(f"  refine                               : {in_opt(model.refine)}")
    print(f"  refine_out                           : {in_opt(model.refine_out)}")
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    print(f"  requires_grad=False 파라미터 ({len(frozen)}): "
          f"{frozen if frozen else '없음 (전부 학습 대상)'}")
    print(HR)

    # =====================================================================
    # [10] grid_sample 좌표 컨벤션 점검 (실제 좌표 캡처)
    # =====================================================================
    print("[10] grid_sample 좌표 컨벤션 (slice_grid 실제 호출 캡처)")
    print(SUB)
    captured = {}
    orig_gs = F.grid_sample

    def _capture_gs(inp, grid, *a, **kw):
        captured["grid"] = grid.detach()
        captured["align_corners"] = kw.get("align_corners", None)
        captured["mode"] = kw.get("mode", None)
        captured["padding_mode"] = kw.get("padding_mode", None)
        captured["input_shape"] = tuple(inp.shape)
        return orig_gs(inp, grid, *a, **kw)

    BG.F.grid_sample = _capture_gs  # 모듈 내 F 참조를 일시 교체 (가중치 불변)
    try:
        with torch.no_grad():
            _ = slice_grid(inter["grid"], gmap)
    finally:
        BG.F.grid_sample = orig_gs  # 원복

    g = captured["grid"]   # (N, D_out, H, W, 3) — 마지막 (x, y, z)
    gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]
    print(f"  grid_sample input(=bilateral grid) shape : {captured['input_shape']}  (N,C,D,gh,gw)")
    print(f"  sample-grid shape : {tuple(g.shape)}   mode={captured['mode']}  "
          f"padding={captured['padding_mode']}  align_corners={captured['align_corners']}")
    print(f"  x(W축) range : [{gx.min().item():+.3f}, {gx.max().item():+.3f}]")
    print(f"  y(H축) range : [{gy.min().item():+.3f}, {gy.max().item():+.3f}]")
    print(f"  z(depth=guidance) range : [{gz.min().item():+.3f}, {gz.max().item():+.3f}]  "
          f"(guidance[0,1]→z[-1,1] 변환 후)")
    z_ok = gz.min().item() >= -1.05 and gz.max().item() <= 1.05 and gz.min().item() < -0.1 + 1.0
    z_full = (gz.max().item() - gz.min().item())
    print(f"  guidance→z 매핑: gz = guidance*2-1 (slice_grid 코드). "
          f"z 사용 폭 = {z_full:.3f} (max 2.0)")
    print(f"  {flag(gz.min().item() >= -1.05 and gz.max().item() <= 1.05, 'z 좌표가 [-1,1] 정상 범위', 'z 좌표 범위 이상')}")
    if z_full < 0.2:
        print(f"       → z 변동폭이 작음: guidance 가 거의 상수라 depth 한 셀만 사용 (slice 고정) [8] 참조")
    print(HR)

    print(" 진단 종료 — 가중치/파일 변경 없음 (read-only).")
    print(HR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
