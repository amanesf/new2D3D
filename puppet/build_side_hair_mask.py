#!/usr/bin/env python3
"""前髪の後れ毛(頬にかかる房)を軽く揺らすための柔らかいマスクを作る。

VIEWER_QUALITY_PLAN.md §1 C 9の再挑戦。線画上、後れ毛はポニーテールの
ような独立した房として区切られておらず頭シルエットと一体化している
ため、輪郭沿いのハードな多角形で切り出すと縁が誤って目・顔にかかる
リスクがある(2026-07-20に一度失敗し保留にした方式)。

今回は方式を変える: 矩形+ガウスぼかしだけの「柔らかい」アルファマスクを
作り、フェザー幅を動かす量より十分大きくとる。境界が線画のどのエッジとも
一致していなくても、ぼかしが広いので継ぎ目が視認できない
(頭シルエットへの軽いシアーと同じ理屈をマスク側でやる)。
"""
import os
import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD = os.path.join(HERE, "layers_zero", "head.png")
OUT = os.path.join(HERE, "layers_zero", "side_hair_mask.png")

# head.png (423x374)内、頬にかかる後れ毛のおおよその矩形。目には掛からない
# よう内側に十分な余白を残す
BOX = (258, 95, 312, 270)
BLUR = 20  # フェザー半径(px)。動かす振幅(数px)よりずっと大きくする


def main():
    head = Image.open(HEAD).convert("RGBA")
    w, h = head.size
    mask = Image.new("L", (w, h), 0)
    box_img = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    d = ImageDraw.Draw(box_img)
    d.rectangle(BOX, fill=255)
    mask = box_img.filter(ImageFilter.GaussianBlur(BLUR))
    # 頭のアルファが無いところ(透明な背景)はマスクも0にしておく
    head_alpha = np.array(head)[:, :, 3]
    m = np.array(mask).astype(np.float32)
    m = np.minimum(m, head_alpha.astype(np.float32))
    Image.fromarray(m.astype(np.uint8), "L").save(OUT)
    print("saved", OUT, "max", m.max())


if __name__ == "__main__":
    main()
