"""세 향상 모델(EnlightenGAN / LUNA / LUNA2-bilateral)의 파라미터 수 비교 (CPU)."""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from src.utils.paths import load_paths

P = load_paths()
PROJ = _ROOT.parent
dev = "cpu"


def count(m):
    return sum(p.numel() for p in m.parameters())


rows = []

# --- LUNA (SmallSizePM generator) ---
try:
    from src.models.luna_base import load_luna_generator
    G = load_luna_generator(Path(str(P.luna_loli30k)), device=dev)
    rows.append(("LUNA (SmallSizePM gen, LoLI-30K)", count(G)))
except Exception as e:
    rows.append(("LUNA", f"ERR {e}"))

# --- LUNA2 (bilateral grid, P1 guidefix) ---
try:
    from src.models.bilateral_grid import build_from_config
    ckpt = _ROOT / "runs" / "bilateral_phase1_l1only_guidefix" / "checkpoints" / "last.pth"
    est = torch.load(str(ckpt), map_location=dev, weights_only=False)
    enh = build_from_config(est["config"])
    rows.append(("LUNA2 (Bilateral-P1 guidefix)", count(enh)))
except Exception as e:
    rows.append(("LUNA2", f"ERR {e}"))

# --- EnlightenGAN generator ---
try:
    egp = PROJ / "EnlightenGAN"
    sys.path.insert(0, str(egp))
    import enlighten_single as eg
    net = eg.load_generator(str(egp / "checkpoints" / "enlightening" / "200_net_G_A.pth"), dev)
    rows.append(("EnlightenGAN (Unet_resize_conv gen)", count(net)))
except Exception as e:
    rows.append(("EnlightenGAN", f"ERR {e}"))

# --- (참고) Zero-DCE / SCI / FUnIE ---
try:
    zp = PROJ / "Zero-DCE" / "Zero-DCE_code"
    sys.path.insert(0, str(zp))
    import model as zdce
    rows.append(("Zero-DCE", count(zdce.enhance_net_nopool())))
except Exception as e:
    rows.append(("Zero-DCE", f"ERR {e}"))
try:
    fp = PROJ / "FUnIE-GAN" / "PyTorch"
    sys.path.insert(0, str(fp))
    from nets.funiegan import GeneratorFunieGAN
    rows.append(("FUnIE-GAN gen", count(GeneratorFunieGAN())))
except Exception as e:
    rows.append(("FUnIE-GAN", f"ERR {e}"))

print("=" * 64)
print(f"  {'Model':<40} {'#Params':>14}")
print("-" * 64)
for name, n in rows:
    if isinstance(n, int):
        print(f"  {name:<40} {n:>14,}  ({n/1e6:.3f} M)")
    else:
        print(f"  {name:<40} {n:>14}")
print("=" * 64)
