#!/usr/bin/env python3
"""視線の虹彩スライド用マスクを作る(VIEWER_QUALITY_PLAN.md §5-1)。

後れ毛(build_side_hair_mask.py)と同じ「輪郭を追わず、円+ガウスぼかし
だけの柔らかいマスクを作り、動かす振幅よりずっと広いフェザーで縁を
吸収する」方式。以前(4df3298)は虹彩だけをハードなワープで動かそうと
して撤去した経緯があるが、今回は境界の精度に依存しないので同じ失敗を
避けられる。
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD = os.path.join(HERE, "layers_zero", "head.png")
OUT_L = os.path.join(HERE, "layers_zero", "iris_mask_l.png")
OUT_R = os.path.join(HERE, "layers_zero", "iris_mask_r.png")

# head.png(423x374)内、両目の中心と半径のおおよその楕円(head.png local座標)
EYE_L = (188, 160, 218, 190)   # 向かって左目(キャラの右目)
EYE_R = (233, 165, 265, 197)   # 向かって右目(キャラの左目)
BLUR = 9  # フェザー半径(px)。スライド振幅(2〜3px)よりずっと広い


def make_mask(box):
    head = Image.open(HEAD).convert("RGBA")
    w, h = head.size
    box_img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(box_img)
    d.ellipse(box, fill=255)
    mask = box_img.filter(ImageFilter.GaussianBlur(BLUR))
    head_alpha = np.array(head)[:, :, 3]
    m = np.minimum(np.array(mask).astype(np.float32), head_alpha.astype(np.float32))
    return Image.fromarray(m.astype(np.uint8), "L")


def main():
    make_mask(EYE_L).save(OUT_L)
    make_mask(EYE_R).save(OUT_R)
    print("saved", OUT_L, OUT_R)


if __name__ == "__main__":
    main()
