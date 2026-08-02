#!/usr/bin/env python3
"""AI生成した表情から目・口パッチだけを切り出してレイヤー登録する。

build_expressions.py の出力(puppet/expressions/raw/)から、採用seedの
画像の目・口領域を元解像度に戻してフェザー付きαで保存し、manifest.jsに
expressionsテーブルを追記する。QC合成画像も出力する。
"""
import json
import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "expressions", "raw")
OUT = os.path.join(HERE, "layers")

# ---- 採用seed(コンタクトシート目視QCで決定) ----
CHOSEN = {"surprised": 1, "joy": 1, "blink": 1, "mouth_aa": 1}

# build_layers.py と一致させること
CROP = (120, 0, 790, 640)
FACE = (420, 60, 660, 300)
GEN_SIZE = 512
SCALE = GEN_SIZE / (FACE[2] - FACE[0])  # 512/240

EYE_L = (460, 150, 516, 200)
EYE_R = (516, 146, 576, 200)
MOUTH = (497, 195, 545, 224)
PAD = 0.30      # パッチは元矩形より30%広めに取る(閉じ目のまつげ等がはみ出すため)
FEATHER = 7     # 境界のフェザー幅px

# 表情ごとにどのパッチを使うか
USE = {
    "surprised": ["eye_l", "eye_r", "mouth"],
    "joy": ["eye_l", "eye_r", "mouth"],
    "blink": ["eye_l", "eye_r"],
    "mouth_aa": ["mouth"],
}
BOXES = {"eye_l": EYE_L, "eye_r": EYE_R, "mouth": MOUTH}


def padded(r):
    w, h = r[2] - r[0], r[3] - r[1]
    px, py = int(w * PAD), int(h * PAD)
    return (r[0] - px, r[1] - py, r[2] + px, r[3] + py)


def feather_alpha(w, h, f):
    a = np.ones((h, w), np.float32)
    ramp = np.linspace(0, 1, f, endpoint=False)
    a[:f, :] *= ramp[:, None]
    a[-f:, :] *= ramp[::-1][:, None]
    a[:, :f] *= ramp[None, :]
    a[:, -f:] *= ramp[::-1][None, :]
    return (a * 255).astype(np.uint8)


with open(os.path.join(HERE, "manifest.js")) as fh:
    manifest = json.loads(fh.read().split("=", 1)[1].rstrip().rstrip(";"))

expr_table = {}
for expr, seed in CHOSEN.items():
    gen = cv2.imread(os.path.join(RAW, f"{expr}_s{seed}.png"), cv2.IMREAD_COLOR)
    entry = {}
    for part in USE[expr]:
        r = padded(BOXES[part])
        # 元座標→生成画像座標
        gx0 = int((r[0] - FACE[0]) * SCALE)
        gy0 = int((r[1] - FACE[1]) * SCALE)
        gx1 = int((r[2] - FACE[0]) * SCALE)
        gy1 = int((r[3] - FACE[1]) * SCALE)
        patch = gen[gy0:gy1, gx0:gx1]
        w, h = r[2] - r[0], r[3] - r[1]
        patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LANCZOS4)
        rgba = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = feather_alpha(w, h, FEATHER)
        name = f"{part}_{expr}"
        cv2.imwrite(os.path.join(OUT, f"{name}.png"), rgba)
        # クロップ座標で登録
        entry[part] = {"layer": name, "x": r[0] - CROP[0], "y": r[1] - CROP[1], "w": w, "h": h}
    expr_table[expr] = entry

manifest["expressions"] = expr_table
with open(os.path.join(HERE, "manifest.js"), "w") as fh:
    fh.write("window.MANIFEST = " + json.dumps(manifest, indent=1) + ";\n")

# QC: base合成の上に各表情パッチを重ねた画像
base = cv2.imread(os.path.join(OUT, "_qc_recomposed.png"))
for expr, entry in expr_table.items():
    qc = base.copy()
    for part, info in entry.items():
        la = cv2.imread(os.path.join(OUT, info["layer"] + ".png"), cv2.IMREAD_UNCHANGED)
        a = la[:, :, 3:4].astype(np.float32) / 255.0
        x, y, w, h = info["x"], info["y"], info["w"], info["h"]
        qc[y:y + h, x:x + w] = (la[:, :, :3] * a + qc[y:y + h, x:x + w] * (1 - a)).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, f"_qc_expr_{expr}.png"), qc)
print("expressions:", json.dumps(expr_table, indent=1))
