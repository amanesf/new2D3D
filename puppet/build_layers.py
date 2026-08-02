#!/usr/bin/env python3
"""silver_ponytail(上半身)をピクセルパペット用レイヤーに分解する。

出力: puppet/layers/*.png + puppet/manifest.js
レイヤー構成(描画順):
  base(ポニーテール抜き) → ponytail → head(目・口inpaint) → eye_l/r, mouth
座標は元画像(1024x1024)ピクセル。出力は上半身クロップ(120,0)-(790,640)。
透明袖(x>345, y>300)越しに見える髪はbase側に残す(揺れの二重描画防止)。
"""
import json
import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "characters", "silver_ponytail", "full.jpg")
OUT = os.path.join(HERE, "layers")
os.makedirs(OUT, exist_ok=True)

img_full = cv2.imread(SRC, cv2.IMREAD_COLOR)
CROP = (120, 0, 790, 640)  # x0,y0,x1,y1 上半身
img = img_full[CROP[1]:CROP[3], CROP[0]:CROP[2]].copy()
H, W = img.shape[:2]
ox, oy = CROP[0], CROP[1]


def P(pts):
    """元画像座標→クロップ座標のポリゴン"""
    return np.array([(x - ox, y - oy) for x, y in pts], np.int32)


# ---- 領域定義(元画像1024座標、QCで微調整) ----
PONYTAIL = P([
    (390, 18), (470, 28), (478, 80), (452, 120), (432, 170), (415, 220),
    (405, 270), (348, 300), (338, 360), (318, 420), (245, 470), (235, 535),
    (238, 590), (205, 638), (150, 640), (128, 592), (152, 545), (198, 490),
    (238, 430), (272, 370), (298, 310), (322, 250), (348, 190), (368, 130),
    (378, 70),
])
HEAD_RECT = (348, 12, 672, 258)     # ヘッドセット・猫耳・マイクブーム込み
EYE_L = (460, 150, 516, 200)
EYE_R = (516, 146, 576, 200)
MOUTH = (497, 195, 545, 224)

ANCHORS = {  # クロップ座標
    "ponytail": (430 - ox, 60 - oy),   # 結び目
    "head": (540 - ox, 252 - oy),      # 首元
    "eye_l": ((460 + 516) // 2 - ox, (150 + 200) // 2 - oy),
    "eye_r": ((516 + 576) // 2 - ox, (146 + 200) // 2 - oy),
    "mouth": ((497 + 545) // 2 - ox, (195 + 224) // 2 - oy),
}

WHITE_THRESH = 244  # 背景は(251,251,251)の薄グレー


def poly_mask(poly):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly], 255)
    return m


def rect_mask(r):
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (r[0] - ox, r[1] - oy), (r[2] - ox, r[3] - oy), 255, -1)
    return m


def remove_border_white(bgr, region_mask):
    """region境界に接する白(背景)連結成分のみ透明化対象にする。内部の白は残す。"""
    white = ((bgr[:, :, 0] >= WHITE_THRESH) & (bgr[:, :, 1] >= WHITE_THRESH)
             & (bgr[:, :, 2] >= WHITE_THRESH) & (region_mask > 0)).astype(np.uint8)
    n, labels = cv2.connectedComponents(white)
    er = cv2.erode(region_mask, np.ones((3, 3), np.uint8))
    edge = cv2.subtract(region_mask, er)
    border_labels = set(np.unique(labels[(edge > 0) & (white > 0)])) - {0}
    return np.isin(labels, list(border_labels)).astype(np.uint8) * 255


def cut_layer(region_mask, name, erase_white=True, src=None):
    s = img if src is None else src
    alpha = region_mask.copy()
    if erase_white:
        alpha[remove_border_white(s, region_mask) > 0] = 0
    rgba = cv2.cvtColor(s, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    ys, xs = np.where(alpha > 0)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cv2.imwrite(os.path.join(OUT, f"{name}.png"), rgba[y0:y1, x0:x1])
    return {"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0)}


mask_pt = poly_mask(PONYTAIL)
mask_head = rect_mask(HEAD_RECT)
mask_head[mask_pt > 0] = 0

layers = {}
layers["ponytail"] = cut_layer(mask_pt, "ponytail")

# 目・口(元画像そのまま、白抜きしない=ハイライト保護)
for name, r in (("eye_l", EYE_L), ("eye_r", EYE_R), ("mouth", MOUTH)):
    layers[name] = cut_layer(rect_mask(r), name, erase_white=False)

# 頭: 目・口をinpaintした画像から切る(まばたき・視線移動時の下地)
inp = np.zeros((H, W), np.uint8)
for r in (EYE_L, EYE_R, MOUTH):
    cv2.rectangle(inp, (r[0] - ox + 2, r[1] - oy + 2), (r[2] - ox - 2, r[3] - oy - 2), 255, -1)
img_head = cv2.inpaint(img, inp, 7, cv2.INPAINT_TELEA)
layers["head"] = cut_layer(mask_head, "head", src=img_head)

# ベース: ポニーテールを消す。背景領域は背景色で塗り、体と重なる帯は横方向最近傍フィル
tt = cv2.dilate(mask_pt, np.ones((5, 5), np.uint8))
yy, xx = np.mgrid[0:H, 0:W]
# 体と重なりうる帯: 透明袖の範囲のみ(元座標x>=340, y 300-520)。それ以外は背景色
body_band = (xx + ox >= 340) & (yy + oy >= 300) & (yy + oy <= 520)
inpaint_mask = (tt > 0) & body_band
bg_mask = (tt > 0) & ~body_band
base = img.copy()
for y in range(H):
    row = inpaint_mask[y]
    if not row.any():
        continue
    xs_ok = np.where(~row)[0]
    xs_ng = np.where(row)[0]
    idx = np.searchsorted(xs_ok, xs_ng).clip(0, len(xs_ok) - 1)
    lo = (idx - 1).clip(0, len(xs_ok) - 1)
    use_lo = np.abs(xs_ng - xs_ok[lo]) < np.abs(xs_ng - xs_ok[idx])
    idx[use_lo] = lo[use_lo]
    base[y, xs_ng] = img[y, xs_ok[idx]]
base[bg_mask] = (251, 251, 251)
# 頭の裏も塗る: 頭レイヤーが動いた時に元の頭が二重に見えるゴースト対策。
# 露出するのは数pxの帯だけなのでTELEAの滲みで十分
head_alpha = np.zeros((H, W), np.uint8)
la = cv2.imread(os.path.join(OUT, "head.png"), cv2.IMREAD_UNCHANGED)
hx, hy = layers["head"]["x"], layers["head"]["y"]
head_alpha[hy:hy + la.shape[0], hx:hx + la.shape[1]] = la[:, :, 3]
head_mask = cv2.dilate((head_alpha > 0).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
base = cv2.inpaint(base, head_mask, 5, cv2.INPAINT_TELEA)
cv2.imwrite(os.path.join(OUT, "base.png"), base)
layers["base"] = {"x": 0, "y": 0, "w": W, "h": H}

manifest = {"size": [W, H], "bg": "#fbfbfb", "layers": layers, "anchors": ANCHORS,
            "draw_order": ["base", "ponytail", "head", "eye_l", "eye_r", "mouth"]}
# 既存manifestのface_bank等(build_face_bank.pyが登録)を保存する
mpath = os.path.join(HERE, "manifest.js")
if os.path.exists(mpath):
    import re
    old = json.loads(re.sub(r"^window\.MANIFEST = |;\s*$", "", open(mpath).read().strip()))
    for k in ("face", "face_bank"):
        if k in old:
            manifest[k] = old[k]
with open(os.path.join(HERE, "manifest.js"), "w") as f:
    f.write("window.MANIFEST = " + json.dumps(manifest, indent=1) + ";\n")

# QC: 静止合成
canvas = base.copy()
for n in ["ponytail", "head", "eye_l", "eye_r", "mouth"]:
    info = layers[n]
    la = cv2.imread(os.path.join(OUT, f"{n}.png"), cv2.IMREAD_UNCHANGED)
    a = la[:, :, 3:4].astype(np.float32) / 255.0
    x, y, w, h = info["x"], info["y"], info["w"], info["h"]
    canvas[y:y + h, x:x + w] = (la[:, :, :3] * a + canvas[y:y + h, x:x + w] * (1 - a)).astype(np.uint8)
cv2.imwrite(os.path.join(OUT, "_qc_recomposed.png"), canvas)

# QC: ポニーテールを+4°回転した時の露出確認
qc2 = base.copy()
Mrot = cv2.getRotationMatrix2D(ANCHORS["ponytail"], -4, 1.0)
info = layers["ponytail"]
la = cv2.imread(os.path.join(OUT, "ponytail.png"), cv2.IMREAD_UNCHANGED)
full = np.zeros((H, W, 4), np.uint8)
full[info["y"]:info["y"] + info["h"], info["x"]:info["x"] + info["w"]] = la
rot = cv2.warpAffine(full, Mrot, (W, H))
a = rot[:, :, 3:4].astype(np.float32) / 255.0
qc2 = (rot[:, :, :3] * a + qc2 * (1 - a)).astype(np.uint8)
for n in ["head", "eye_l", "eye_r", "mouth"]:
    info = layers[n]
    la = cv2.imread(os.path.join(OUT, f"{n}.png"), cv2.IMREAD_UNCHANGED)
    a = la[:, :, 3:4].astype(np.float32) / 255.0
    x, y, w, h = info["x"], info["y"], info["w"], info["h"]
    qc2[y:y + h, x:x + w] = (la[:, :, :3] * a + qc2[y:y + h, x:x + w] * (1 - a)).astype(np.uint8)
cv2.imwrite(os.path.join(OUT, "_qc_sway.png"), qc2)
print("layers:", layers)
print("done")
