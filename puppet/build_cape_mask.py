#!/usr/bin/env python3
"""ケープ裾の先端だけを軽く揺らすための柔らかいマスクを作る。

VIEWER_QUALITY_PLAN.md §5「動きの総量が構造的に小さい」対策の一つ。
後れ毛(build_side_hair_mask.py)と同じ「矩形+ガウスぼかしだけの柔らかい
マスク」方式。今回のポニーテール根元装飾の教訓(2026-07-20 その3)を
踏まえ、対象範囲は意図的に体・脚・ストラップ等の造形にかからない
ケープ裾の先端(画面右下、無地の布地のみ)に限定した
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "layers_zero", "base.png")
OUT = os.path.join(HERE, "layers_zero", "cape_hem_mask.png")

# base.png(512x704)内、脚・ストラップに掛からないケープ裾先端のみの矩形
BOX = (370, 540, 500, 704)
BLUR = 40  # フェザー半径(px)。動かす振幅(数px)よりずっと大きくする


def main():
    base = Image.open(BASE).convert("RGBA")
    w, h = base.size
    box_img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(box_img)
    d.rectangle(BOX, fill=255)
    mask = box_img.filter(ImageFilter.GaussianBlur(BLUR))
    base_alpha = np.array(base)[:, :, 3]
    m = np.minimum(np.array(mask).astype(np.float32), base_alpha.astype(np.float32))
    Image.fromarray(m.astype(np.uint8), "L").save(OUT)
    print("saved", OUT, "max", m.max())


if __name__ == "__main__":
    main()
