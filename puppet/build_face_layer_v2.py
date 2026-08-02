#!/usr/bin/env python3
"""zeroキャラ新マスター(characters/zero/master_v2_fullbody.png)の
レイヤー分解(v2、ポーズ再設計後)。

build_face_layer.py(旧master_bust.png向け)の後継。新マスターは
ポニーテールが体・腕と一切重ならない構図(SUPER_LIVE2D_V3_PLAN.md
Spike S1の教訓を反映)なので、旧版で必要だった「ponytailとheadの
重なり除去」処理が不要になり、分解がシンプルになった。

VIEWER_QUALITY_PLAN.md §3の方針(座標のハードコード全廃・
マニフェストJSON化)を踏まえ、座標定義はこのファイル内の定数ではなく
manifest_v2.json に出力し、他スクリプト(将来のレンダラ・QC)は
そのJSONだけを読む前提とする。

構成: base(無傷のフル原画) → ponytail(縁フェザー) → head(縁フェザー)
"""
import json
import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "characters", "zero", "master_v2_fullbody.png")
OUT = os.path.join(HERE, "layers_zero_v2")
os.makedirs(OUT, exist_ok=True)

img_full = cv2.imread(SRC, cv2.IMREAD_COLOR)
H, W = img_full.shape[:2]  # 704, 512

# 顔差し替え対象(目・眉・口・頬)。新マスターは全身構図のため顔が
# 旧HEAD_RECTより小さい領域になる
FACE_BOX = (192, 138, 308, 232)
# head: 髪上端・猫耳・ヘッドセット(ブーム両側含む)・顔・首元まで
HEAD_RECT = (150, 92, 340, 262)
# ponytail: 頭から右へ流れる毛束。新構図では体・腕と非接触なので、
# 輪郭ポリゴンはheadとの重なり回避だけを考えればよい(簡素化)
PONYTAIL_POLY = [
    (298, 103), (325, 106), (365, 120), (410, 143), (445, 173), (468, 203),
    (480, 233), (470, 262), (447, 287), (417, 308), (382, 325), (350, 340),
    (315, 338), (330, 315), (345, 295), (355, 272), (350, 248), (335, 225),
    (315, 203), (300, 180), (288, 158), (280, 135), (280, 115),
]
PONYTAIL_ANCHOR = (300, 115)  # 頭への接続点(master座標)

WHITE_THRESH = 244
FEATHER_PX = 9


def rect_mask(r, shape=(H, W)):
    m = np.zeros(shape, np.uint8)
    cv2.rectangle(m, (r[0], r[1]), (r[2], r[3]), 255, -1)
    return m


def poly_mask(poly, shape=(H, W)):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
    return m


def remove_border_white(bgr, region_mask):
    white = ((bgr[:, :, 0] >= WHITE_THRESH) & (bgr[:, :, 1] >= WHITE_THRESH)
             & (bgr[:, :, 2] >= WHITE_THRESH) & (region_mask > 0)).astype(np.uint8)
    n, labels = cv2.connectedComponents(white)
    er = cv2.erode(region_mask, np.ones((3, 3), np.uint8))
    edge = cv2.subtract(region_mask, er)
    border_labels = set(np.unique(labels[(edge > 0) & (white > 0)])) - {0}
    return np.isin(labels, list(border_labels)).astype(np.uint8) * 255


def feathered_alpha(bgr, region_mask):
    alpha = region_mask.copy()
    alpha[remove_border_white(bgr, region_mask) > 0] = 0
    k = FEATHER_PX * 2 + 1
    return cv2.GaussianBlur(alpha, (k, k), 0)


# ---- ponytail ----
ponytail_mask_full = poly_mask(PONYTAIL_POLY)
pys, pxs = np.where(ponytail_mask_full > 0)
px0, px1, py0, py1 = pxs.min(), pxs.max() + 1, pys.min(), pys.max() + 1
ponytail_alpha = feathered_alpha(img_full, ponytail_mask_full)
ponytail_rgba = cv2.cvtColor(img_full, cv2.COLOR_BGR2BGRA)
ponytail_rgba[:, :, 3] = ponytail_alpha
px0, py0 = max(0, px0 - FEATHER_PX), max(0, py0 - FEATHER_PX)
px1, py1 = min(W, px1 + FEATHER_PX), min(H, py1 + FEATHER_PX)
ponytail_crop = ponytail_rgba[py0:py1, px0:px1]
cv2.imwrite(os.path.join(OUT, "ponytail.png"), ponytail_crop)

# ---- head(ponytailと重なる領域は除いて切り出す) ----
head_mask = rect_mask(HEAD_RECT)
head_mask[ponytail_mask_full > 0] = 0
head_alpha = feathered_alpha(img_full, head_mask)
head_rgba = cv2.cvtColor(img_full, cv2.COLOR_BGR2BGRA)
head_rgba[:, :, 3] = head_alpha
hx0, hy0, hx1, hy1 = HEAD_RECT
hx0, hy0 = max(0, hx0 - FEATHER_PX), max(0, hy0 - FEATHER_PX)
hx1, hy1 = min(W, hx1 + FEATHER_PX), min(H, hy1 + FEATHER_PX)
head_crop = head_rgba[hy0:hy1, hx0:hx1]
cv2.imwrite(os.path.join(OUT, "head.png"), head_crop)

# ---- face フェザーマスク(head crop内の相対座標) ----
fx0, fy0, fx1, fy1 = FACE_BOX
rel = (fx0 - hx0, fy0 - hy0, fx1 - hx0, fy1 - hy0)
feather_mask = np.zeros((hy1 - hy0, hx1 - hx0), np.uint8)
cv2.rectangle(feather_mask, (rel[0], rel[1]), (rel[2], rel[3]), 255, -1)
k = 21
feather_mask = cv2.GaussianBlur(feather_mask, (k, k), 0)
cv2.imwrite(os.path.join(OUT, "face_feather_mask.png"), feather_mask)

# ---- base(無傷のフル原画) ----
cv2.imwrite(os.path.join(OUT, "base.png"), img_full)

# ---- 合成QC ----
canvas = cv2.cvtColor(img_full, cv2.COLOR_BGR2BGRA)


def alpha_paste(dst, src, x0, y0):
    h, w = src.shape[:2]
    roi = dst[y0:y0 + h, x0:x0 + w]
    a = (src[:, :, 3:4].astype(np.float32) / 255.0)
    roi[:, :, :3] = (roi[:, :, :3].astype(np.float32) * (1 - a)
                      + src[:, :, :3].astype(np.float32) * a).astype(np.uint8)
    roi[:, :, 3] = np.maximum(roi[:, :, 3], src[:, :, 3])
    dst[y0:y0 + h, x0:x0 + w] = roi


alpha_paste(canvas, head_crop, hx0, hy0)
alpha_paste(canvas, ponytail_crop, px0, py0)
cv2.imwrite(os.path.join(OUT, "_qc_recomposed.png"), canvas)

# ---- マニフェスト(宣言データ化。VIEWER_QUALITY_PLAN §3の本丸への一歩) ----
manifest = {
    "character": "zero_v2",
    "master": "characters/zero/master_v2_fullbody.png",
    "canvas": {"w": W, "h": H},
    "layers": {
        "base": {"file": "base.png", "kind": "static_full"},
        "ponytail": {
            "file": "ponytail.png", "kind": "feathered_cutout",
            "crop": [int(px0), int(py0), int(px1), int(py1)],
            "polygon_master_space": PONYTAIL_POLY,
            "anchor_master_space": list(PONYTAIL_ANCHOR),
            "feather_px": FEATHER_PX,
        },
        "head": {
            "file": "head.png", "kind": "feathered_cutout",
            "crop": [int(hx0), int(hy0), int(hx1), int(hy1)],
            "rect_master_space": list(HEAD_RECT),
            "feather_px": FEATHER_PX,
            "face_feather_mask": "face_feather_mask.png",
            "face_box_master_space": list(FACE_BOX),
        },
    },
    "notes": [
        "旧master_bust.pngと異なり、ponytailは体・腕と非接触の構図"
        "(SUPER_LIVE2D_V3_PLAN.md Spike S1参照)。",
        "expr_adopted/viseme_candidatesは旧マスターの顔スケールで生成"
        "されているため、face_box_master_spaceのサイズに合わせて要再生成"
        "(単純な貼り替えは不可)。",
    ],
}
with open(os.path.join(OUT, "manifest_v2.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("done", W, H)
