"""[P2 준비 1] ExDark train/val/test split 생성 + 통계 (잠정).

배경
----
ExDark README 는 ``annotations/imageclasslist.txt`` 에 공식 split(1/2/3)이 있다고
하지만 이 사본에는 **그 파일이 없다**. 따라서 사용자 지시에 따라 **표준에 준한
잠정 split** 을 결정론적으로 생성한다. (공식 아님 — GPT 확정 시 교체.)

방식
----
* 클래스별(EXDARK 12) stratified, 파일명 사전순 정렬 후 결정론적 분할.
* 비율 train:val:test = 70:10:20 (재현 가능, seed 불요).
* 데이터셋 디렉토리는 수정하지 않고 ``configs/exdark_split_provisional.csv`` 에 저장.
  포맷: ``image_name,class_dir,split`` (split ∈ {1=train,2=val,3=test}).

GT 가 0개인 이미지도 split 에는 포함(평가 동일 조건). 통계에 GT box 수도 보고.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

_LUNA2_ROOT = Path(__file__).resolve().parent.parent
if str(_LUNA2_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUNA2_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

from experiments.eval_detection import (
    EXDARK_CLASSES, collect_exdark_samples, parse_bbgt_v3,
)
from src.utils.paths import load_paths

HR = "=" * 84
RATIO = (0.70, 0.10, 0.20)  # train, val, test (잠정)
OUT_CSV = _LUNA2_ROOT / "configs" / "exdark_split_provisional.csv"


def assign_split(n: int):
    """길이 n 정렬 리스트의 인덱스별 split(1/2/3) 배열 (결정론적 70/10/20)."""
    n_tr = int(round(n * RATIO[0]))
    n_va = int(round(n * RATIO[1]))
    out = []
    for i in range(n):
        if i < n_tr:
            out.append(1)
        elif i < n_tr + n_va:
            out.append(2)
        else:
            out.append(3)
    return out


def main() -> int:
    P = load_paths()
    print(HR)
    print(" [P2-1] ExDark 잠정 split 생성 (공식 파일 부재 → 표준 70/10/20, 결정론적)")
    print(HR)
    print(f"  exdark_root : {P.exdark}")

    # 전체 샘플 (split 필터 없이) — 클래스별 그룹
    samples = collect_exdark_samples(P.exdark, splits=None)
    by_class = defaultdict(list)
    for sm in samples:
        by_class[sm.class_dir].append(sm)

    rows = []                       # (image_name, class_dir, split)
    # 통계 누적
    img_cnt = {1: 0, 2: 0, 3: 0}
    gt_cnt = {1: 0, 2: 0, 3: 0}
    per_class = defaultdict(lambda: {1: 0, 2: 0, 3: 0})

    for cls in EXDARK_CLASSES:
        sms = sorted(by_class.get(cls, []), key=lambda s: s.image_path.name)
        splits = assign_split(len(sms))
        for sm, sp in zip(sms, splits):
            n_gt = len(parse_bbgt_v3(sm.ann_path))
            rows.append((sm.image_path.name, sm.class_dir, sp))
            img_cnt[sp] += 1
            gt_cnt[sp] += n_gt
            per_class[cls][sp] += 1

    # CSV 저장 (데이터셋 아닌 LUNA2/configs)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "class_dir", "split"])  # split: 1=train,2=val,3=test
        w.writerows(rows)

    total = len(rows)
    print(f"  총 이미지   : {total}")
    print("-" * 84)
    print(f"  {'split':<8} | {'images':>7} ({'%':>5}) | {'GT boxes':>9}")
    print("-" * 84)
    names = {1: "train", 2: "val", 3: "test"}
    for sp in (1, 2, 3):
        print(f"  {names[sp]:<8} | {img_cnt[sp]:>7} ({img_cnt[sp]/total*100:>4.1f}%) | {gt_cnt[sp]:>9}")
    print("-" * 84)
    print("  클래스별 이미지 분포 (train/val/test):")
    print(f"  {'class':<12} | {'train':>6} {'val':>5} {'test':>6} | {'total':>6}")
    for cls in EXDARK_CLASSES:
        c = per_class[cls]
        tot = c[1] + c[2] + c[3]
        print(f"  {cls:<12} | {c[1]:>6} {c[2]:>5} {c[3]:>6} | {tot:>6}")
    print(HR)
    print(f"  저장 → {OUT_CSV}")
    print("  ⚠ 공식 split 아님 (잠정). GPT 확정 시 이 CSV 교체로 일괄 반영됨.")
    print(HR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
