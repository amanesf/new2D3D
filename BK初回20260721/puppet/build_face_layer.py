#!/usr/bin/env python3
"""zeroキャラ(characters/zero/master_bust.png)を顔差し替え可能な
head/faceレイヤーに分解する(精密レイヤー分解、顔領域スコープ)。

VIEWER_QUALITY_PLAN.md §1-2の「四角い枠・切れ目」対策(Live2D式重ね)を
反映した版:
- baseは**無傷のフル原画**(穴を開けない・埋めない)。矩形穴+色フィルの
  構造的限界(背景が非均一で色が合わない/head矩形の下端が腕を横切る)を
  そもそも発生させない。
- head/ponytailは輪郭沿いのアルファ抽出+縁数pxのフェザー(ガウスぼかし)
  で切り出す。baseが無傷なので、動いた時に露出するのは「同じ絵の縁」で
  あり、フェザーで継ぎ目が視認できなくなる(Live2Dのアートメッシュと
  同じ理屈)。

構成:
  base(無傷のフル原画) → ponytail(縁フェザー) → head(縁フェザー) →
  face(表情/viseme差し替え用、headの内側にフェザーマスクで合成する固定領域)

face領域はgen_expr.py/gen_viseme.pyのFACE_BOXと同一座標系
(master座標(80,20)-(432,340))なので、expr_adopted/*.pngをそのまま
差し替え候補として使える。
"""
import json
import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "characters", "zero", "master_bust.png")
OUT = os.path.join(HERE, "layers_zero")
os.makedirs(OUT, exist_ok=True)

img_full = cv2.imread(SRC, cv2.IMREAD_COLOR)
H, W = img_full.shape[:2]  # 704, 512

# gen_expr.py / gen_viseme.py の FACE_BOX と同一(表情差し替え対象)
FACE_BOX = (80, 20, 432, 340)
# head: 髪上端・猫耳・ヘッドセット・顔・首元まで含む矩形(rigid layer)
HEAD_RECT = (55, 0, 460, 365)
# ponytail: 頭から垂れる毛束(head_rectの下まで続くので別レイヤーにしないと
# 頭が動いた時に継ぎ目が生まれる)。頭との接続点付近をアンカーにして
# headと一緒に(または追従ばねで少し遅れて)動かす
PONYTAIL_POLY = [
    (150, 42), (195, 52), (215, 90), (208, 140), (196, 190), (178, 230),
    (155, 265), (130, 295), (108, 325), (90, 355), (75, 385), (62, 415),
    (50, 445), (38, 475), (28, 505), (18, 535), (8, 565), (0, 590),
    (0, 530), (2, 470), (6, 410), (10, 350), (12, 290), (12, 235),
    (16, 185), (28, 145), (48, 105), (72, 78), (105, 58),
]
PONYTAIL_ANCHOR = (170, 60)  # 頭への接続点(master座標)

WHITE_THRESH = 244


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


FEATHER_PX = 9  # 縁フェザー幅(奇数のガウスカーネル半径相当)


def feathered_alpha(bgr, region_mask):
    """背景に接する縁はremove_border_whiteでクリスプに透明化しつつ、
    それ以外の縁(体を横切る切り口等)は数pxフェザーして継ぎ目を消す。
    baseが無傷のフル原画なので、フェザーで露出するのは同じ絵の縁になる。"""
    alpha = region_mask.copy()
    alpha[remove_border_white(bgr, region_mask) > 0] = 0
    k = FEATHER_PX * 2 + 1
    return cv2.GaussianBlur(alpha, (k, k), 0)


# ---- ponytail レイヤー(頭から続く毛束。headと別レイヤーにして
#      同じtransformで動かすことで、頭部変形時の継ぎ目を防ぐ) ----
ponytail_mask_full = poly_mask(PONYTAIL_POLY)
pys, pxs = np.where(ponytail_mask_full > 0)
px0, px1, py0, py1 = pxs.min(), pxs.max() + 1, pys.min(), pys.max() + 1
ponytail_alpha = feathered_alpha(img_full, ponytail_mask_full)
ponytail_rgba = cv2.cvtColor(img_full, cv2.COLOR_BGR2BGRA)
ponytail_rgba[:, :, 3] = ponytail_alpha
# フェザーがcropの外にはみ出さないよう、bboxをFEATHER_PX分広げる
px0, py0 = max(0, px0 - FEATHER_PX), max(0, py0 - FEATHER_PX)
px1, py1 = min(W, px1 + FEATHER_PX), min(H, py1 + FEATHER_PX)
ponytail_crop = ponytail_rgba[py0:py1, px0:px1]
cv2.imwrite(os.path.join(OUT, "ponytail.png"), ponytail_crop)

# ---- head レイヤー(ponytailと重なる部分は除いて切り出す。背景は透明化) ----
head_mask = rect_mask(HEAD_RECT)
head_mask[ponytail_mask_full > 0] = 0  # ponytailは別レイヤーなので穴をあけておく
head_alpha = feathered_alpha(img_full, head_mask)
head_rgba = cv2.cvtColor(img_full, cv2.COLOR_BGR2BGRA)
head_rgba[:, :, 3] = head_alpha
hx0, hy0, hx1, hy1 = HEAD_RECT
# フェザーがcropの外にはみ出さないよう、bboxをFEATHER_PX分広げる
hx0, hy0 = max(0, hx0 - FEATHER_PX), max(0, hy0 - FEATHER_PX)
hx1, hy1 = min(W, hx1 + FEATHER_PX), min(H, hy1 + FEATHER_PX)
head_crop = head_rgba[hy0:hy1, hx0:hx1]
cv2.imwrite(os.path.join(OUT, "head.png"), head_crop)

# ---- face フェザーマスク(head crop内の相対座標、表情差し替え合成用) ----
fx0, fy0, fx1, fy1 = FACE_BOX
rel = (fx0 - hx0, fy0 - hy0, fx1 - hx0, fy1 - hy0)
feather_mask = np.zeros((hy1 - hy0, hx1 - hx0), np.uint8)
cv2.rectangle(feather_mask, (rel[0], rel[1]), (rel[2], rel[3]), 255, -1)
feather_mask = cv2.GaussianBlur(feather_mask, (31, 31), 0)
cv2.imwrite(os.path.join(OUT, "face_feather_mask.png"), feather_mask)
# ブラウザのdestination-in合成用にRGBA化(alpha=グレースケール値)
mask_rgba = np.zeros((*feather_mask.shape, 4), np.uint8)
mask_rgba[:, :, 3] = feather_mask
cv2.imwrite(os.path.join(OUT, "face_feather_mask_rgba.png"), mask_rgba)

# ---- base レイヤー(無傷のフル原画。穴は開けない・埋めない) ----
# VIEWER_QUALITY_PLAN.md §1-2: 矩形穴+色フィルは背景の非均一さで色が
# 合わず「枠」の原因になっていた。baseを無傷にすれば、head/ponytailの
# フェザー縁が動いても露出するのは常に「同じ絵」なので継ぎ目が出ない。
base = img_full.copy()
bg_color = tuple(int(v) for v in img_full[5, 5])
cv2.imwrite(os.path.join(OUT, "base.png"), base)

manifest = {
    "size": [W, H],
    "bg": "#%02x%02x%02x" % (bg_color[2], bg_color[1], bg_color[0]),
    "head": {"x": hx0, "y": hy0, "w": hx1 - hx0, "h": hy1 - hy0},
    "face_hole": {"x": rel[0], "y": rel[1], "w": rel[2] - rel[0], "h": rel[3] - rel[1]},
    "ponytail": {"x": int(px0), "y": int(py0), "w": int(px1 - px0), "h": int(py1 - py0)},
    "ponytail_anchor": {"x": PONYTAIL_ANCHOR[0], "y": PONYTAIL_ANCHOR[1]},
    "draw_order": ["base", "ponytail", "head", "face"],
}
with open(os.path.join(OUT, "manifest_face.json"), "w") as f:
    json.dump(manifest, f, indent=1, ensure_ascii=False)

# ---- QC: baseにponytail+headを貼り戻して原画と一致するか確認 ----
canvas = base.copy()
ap = (ponytail_crop[:, :, 3:4].astype(np.float32) / 255.0)
canvas[py0:py1, px0:px1] = (ponytail_crop[:, :, :3] * ap + canvas[py0:py1, px0:px1] * (1 - ap)).astype(np.uint8)
a = (head_crop[:, :, 3:4].astype(np.float32) / 255.0)
canvas[hy0:hy1, hx0:hx1] = (head_crop[:, :, :3] * a + canvas[hy0:hy1, hx0:hx1] * (1 - a)).astype(np.uint8)
cv2.imwrite(os.path.join(OUT, "_qc_recomposed.png"), canvas)

# ---- QC: neutralをjoyに差し替えた合成 ----
expr_dir = os.path.join(HERE, "..", "characters", "zero", "expr_adopted")
joy = cv2.imread(os.path.join(expr_dir, "joy.png"), cv2.IMREAD_COLOR)
joy = cv2.resize(joy, (rel[2] - rel[0], rel[3] - rel[1]))
qc2 = canvas.copy()
fm = feather_mask[rel[1]:rel[3], rel[0]:rel[2]].astype(np.float32) / 255.0
fm3 = fm[:, :, None]
region = qc2[hy0 + rel[1]:hy0 + rel[3], hx0 + rel[0]:hx0 + rel[2]]
qc2[hy0 + rel[1]:hy0 + rel[3], hx0 + rel[0]:hx0 + rel[2]] = (joy * fm3 + region * (1 - fm3)).astype(np.uint8)
cv2.imwrite(os.path.join(OUT, "_qc_swap_joy.png"), qc2)

print("done", manifest)
