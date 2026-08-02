#!/usr/bin/env python3
"""呼吸で胸・肋骨まわりだけがわずかに膨らむ表現用の柔らかいマスクを作る。

VIEWER_QUALITY_PLAN.md §5-3。これまでの呼吸(P.bodyY)は体全体の平行移動
のみで、Live2D特有の「体が膨らむ」変形が無かった。腕・脚まで一律に
スケールするとおかしくなるため、後れ毛・ケープ裾と同じソフトマスク方式
(矩形+ガウスぼかし)で胸・肋骨まわりだけを切り出し、その部分だけを
胴の下端を支点にわずかにスケールさせる
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "layers_zero", "base.png")
OUT = os.path.join(HERE, "layers_zero", "chest_mask.png")

# base.png(512x704)内、胸・肋骨まわり(腕・脚にはなるべくかからない)
BOX = (185, 265, 315, 405)
BLUR = 30  # フェザー半径(px)。動かすスケール量(1.0前後)による移動量よりずっと大きい


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
