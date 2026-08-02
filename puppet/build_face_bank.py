#!/usr/bin/env python3
"""v2設計: 顔内側領域まるごと差し替えのキーフレームバンク構築。

役割:
  --prep     : img2img入力(顔クロップ512px)とフェザーマスクを生成
  --register : expressions/raw/ の採用候補を顔キーフレームPNG(小サイズ)に変換し
               puppet/face/ へ登録、manifest.js の face_bank を更新
  --score    : raw候補をフェザー境界帯の一致度で機械採点(選別の下読み)

顔領域は元画像座標 FACE=(420,60,660,300) の240x240固定。キーフレームは
240x240 RGBA(フェザーマスク焼き込み)なのでファイルは小さい。
"""
import argparse
import glob
import json
import os
import re
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "characters", "silver_ponytail", "full.jpg")
RAW = os.path.join(HERE, "expressions", "raw")
FACE_DIR = os.path.join(HERE, "face")
MANIFEST = os.path.join(HERE, "manifest.js")

FACE = (420, 60, 660, 300)          # 元画像座標
CROP_OX, CROP_OY = 120, 0           # build_layers.py の上半身クロップ原点
FW, FH = FACE[2] - FACE[0], FACE[3] - FACE[1]
GEN_SIZE = 512
FEATHER = 22                        # フェザー幅(px, 240px系)


def feather_mask(kind="face"):
    """部位別フェザーマスク(FACE 240px座標)。差し替えは必要最小の領域だけにする:
    face=眉〜顎の楕円(表情全体) / eyes=両目帯(blink用) / mouth=口まわり(viseme用)。
    髪・ヘッドセット・輪郭は常に原画のまま残す。"""
    m = np.zeros((FH, FW), np.float32)
    if kind == "face":
        cv2.ellipse(m, (117, 124), (45, 39), 0, 0, 360, 1.0, -1)
    elif kind == "eyes":
        cv2.ellipse(m, (88, 116), (26, 20), 0, 0, 360, 1.0, -1)
        cv2.ellipse(m, (150, 114), (26, 20), 0, 0, 360, 1.0, -1)
    elif kind == "mouth":
        cv2.ellipse(m, (117, 152), (26, 18), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 5 if kind != "face" else 7)
    return m


MASK_KIND = {"blink": "eyes", "mouth_aa": "mouth"}  # 既定はface


def load_face_src():
    img = cv2.imread(SRC, cv2.IMREAD_COLOR)
    return img[FACE[1]:FACE[3], FACE[0]:FACE[2]]


def prep():
    os.makedirs(FACE_DIR, exist_ok=True)
    face = load_face_src()
    cv2.imwrite(os.path.join(FACE_DIR, "_source_512.png"),
                cv2.resize(face, (GEN_SIZE, GEN_SIZE), interpolation=cv2.INTER_LANCZOS4))
    m = (feather_mask() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(FACE_DIR, "_feather_mask.png"), m)
    print(f"prep done: source 512px + feather mask ({FW}x{FH}, feather={FEATHER})")


def border_score(cand240, ref240, mask):
    """フェザー境界帯(0<mask<1)での差の小ささを採点。高いほど馴染む。"""
    band = (mask > 0.05) & (mask < 0.95)
    d = np.abs(cand240.astype(np.float32) - ref240.astype(np.float32)).mean(axis=2)
    return float(100 - d[band].mean())


CASCADE = os.path.join(HERE, "lbpcascade_animeface.xml")
# 原画full.jpg上の顔箱(カスケード検出値、固定)と顔領域FACEの相対関係が基準
SRC_BOX = (516, 197, 72)  # x, y, w


def _detect_face(im, mn=3):
    c = cv2.CascadeClassifier(CASCADE)
    g = cv2.equalizeHist(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    r = c.detectMultiScale(g, 1.05, mn, minSize=(40, 40))
    if len(r) == 0:
        return None
    return max(r, key=lambda b: b[2])  # 最大の顔箱


def align_to_source(cand):
    """アニメ顔カスケードの顔箱同士の相似変換で、候補からFACE領域相当の窓を
    切り出す。仕上げに目領域テンプレートのNCCで平行移動を微修正。"""
    box = _detect_face(cand)
    if box is None:
        return cv2.resize(cand, (FW, FH), interpolation=cv2.INTER_AREA), -1.0, 1.0
    bx, by, bw = int(box[0]), int(box[1]), int(box[2])
    k0 = bw / float(SRC_BOX[2])  # 候補px / 原画px(粗い。カスケード箱は揺れる)
    # スケール込みNCC微修正: 前髪〜目の上半分帯(表情の影響が小さい)を
    # テンプレートに、スケールk0*[0.55..1.35]×全域探索で最良を取る
    src = load_face_src()
    band = cv2.cvtColor(src[20:150, 30:210], cv2.COLOR_BGR2GRAY)  # FACE座標系
    cg = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
    best = (-2, k0, 0, 0)
    for s in np.linspace(k0 * 0.55, k0 * 1.35, 17):
        t = cv2.resize(band, None, fx=s, fy=s)
        if t.shape[0] >= cg.shape[0] or t.shape[1] >= cg.shape[1] or t.shape[0] < 20:
            continue
        r = cv2.matchTemplate(cg, t, cv2.TM_CCOEFF_NORMED)
        _, v, _, (lx, ly) = cv2.minMaxLoc(r)
        if v > best[0]:
            best = (v, s, lx, ly)
    v, k, lx, ly = best
    x0 = int(lx - 30 * k)
    y0 = int(ly - 20 * k)
    w = int(FW * k)
    pad = max(0, -x0, -y0, x0 + w - cand.shape[1], y0 + w - cand.shape[0])
    if pad:
        cand = cv2.copyMakeBorder(cand, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0 = x0 + pad, y0 + pad
    win = cand[y0:y0 + w, x0:x0 + w]
    return cv2.resize(win, (FW, FH), interpolation=cv2.INTER_AREA), float(v), k


def to_keyframe(raw_path):
    cand = cv2.imread(raw_path, cv2.IMREAD_COLOR)
    if "_c" in os.path.basename(raw_path):  # 第2次以降: 位置合わせあり
        kf, v, s = align_to_source(cand)
        return kf
    return cv2.resize(cand, (FW, FH), interpolation=cv2.INTER_AREA)


def score():
    ref = load_face_src()
    mask = feather_mask()
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, "*.png"))):
        n = os.path.basename(p)
        if n.startswith("_"):
            continue
        rows.append((border_score(to_keyframe(p), ref, mask), n))
    for s, n in sorted(rows, reverse=True):
        print(f"{s:6.2f}  {n}")


def align_sheet():
    """第2次raw(*_c*.jpg)を位置合わせ+フェザー合成した状態でコンタクトシート化。
    実際に貼った見た目で選別できる。"""
    ref = load_face_src()
    mask = feather_mask()[:, :, None]
    by_target = {}
    for p in sorted(glob.glob(os.path.join(RAW, "*_c*.jpg"))):
        n = os.path.basename(p)
        t = n.rsplit("_c", 1)[0]
        kf, v, s = align_to_source(cv2.imread(p, cv2.IMREAD_COLOR))
        blend = (kf * mask + ref * (1 - mask)).astype(np.uint8)
        tile = cv2.resize(blend, (200, 200))
        cv2.putText(tile, n[len(t) + 1:-4], (4, 20), 0, 0.55, (0, 0, 255), 2)
        cv2.putText(tile, f"m{v:.2f}", (4, 195), 0, 0.5, (255, 0, 0), 2)
        by_target.setdefault(t, []).append(tile)
    for t, tiles in by_target.items():
        rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
        out = os.path.join(HERE, "expressions", f"_sheet2_{t}.jpg")
        cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("wrote", out)


def register_wink_mirror(key, raw, closed_side):
    """ウインク候補の閉じ目側半分をミラーして両目閉じキーフレームを合成。
    closed_side: 閉じ目がある側('l'=画像左半分)。"""
    os.makedirs(FACE_DIR, exist_ok=True)
    kf, v, s = align_to_source(cv2.imread(os.path.join(RAW, raw + ".jpg")))
    flip = cv2.flip(kf, 1)
    half = np.zeros((FH, FW), np.float32)
    half[:, :FW // 2] = 1.0
    half = cv2.GaussianBlur(half, (0, 0), 10)
    if closed_side == "l":
        out = kf * half[:, :, None] + flip * (1 - half[:, :, None])
    else:
        out = flip * half[:, :, None] + kf * (1 - half[:, :, None])
    rgba = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = (feather_mask(MASK_KIND.get(key, "face")) * 255).astype(np.uint8)
    path = os.path.join(FACE_DIR, key + ".png")
    cv2.imwrite(path, rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    with open(MANIFEST) as f:
        man = json.loads(re.sub(r"^window\.MANIFEST = |;\s*$", "", f.read().strip()))
    man.setdefault("face_bank", {})[key] = "face/" + key + ".png"
    man["face"] = {"x": FACE[0] - CROP_OX, "y": FACE[1] - CROP_OY, "w": FW, "h": FH}
    with open(MANIFEST, "w") as f:
        f.write("window.MANIFEST = " + json.dumps(man, indent=1) + ";\n")
    print(f"registered {key} <- {raw} (wink mirror {closed_side}, m={v:.2f})")


def register(names):
    """names: 'blink=blink_b65_s3' 形式。rawを240pxキーフレーム化して登録。"""
    os.makedirs(FACE_DIR, exist_ok=True)
    with open(MANIFEST) as f:
        man = json.loads(re.sub(r"^window\.MANIFEST = |;\s*$", "", f.read().strip()))
    bank = man.setdefault("face_bank", {})
    man["face"] = {"x": FACE[0] - CROP_OX, "y": FACE[1] - CROP_OY, "w": FW, "h": FH}
    for spec in names:
        key, raw = spec.split("=")
        cands = [p for e in (".png", ".jpg") if os.path.exists(p := os.path.join(RAW, raw + e))]
        kf = to_keyframe(cands[0])
        rgba = cv2.cvtColor(kf, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = (feather_mask(MASK_KIND.get(key, "face")) * 255).astype(np.uint8)
        out = os.path.join(FACE_DIR, key + ".png")
        cv2.imwrite(out, rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        bank[key] = "face/" + key + ".png"
        print(f"registered {key} <- {raw} ({os.path.getsize(out)//1024}KB)")
    with open(MANIFEST, "w") as f:
        f.write("window.MANIFEST = " + json.dumps(man, indent=1) + ";\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--align-sheet", action="store_true")
    ap.add_argument("--register", nargs="+", metavar="key=rawname")
    ap.add_argument("--register-wink", nargs=3, metavar=("key", "rawname", "side"))
    a = ap.parse_args()
    if a.register_wink:
        register_wink_mirror(*a.register_wink)
    if a.prep:
        prep()
    if a.align_sheet:
        align_sheet()
    if a.score:
        score()
    if a.register:
        register(a.register)
