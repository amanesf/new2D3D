#!/usr/bin/env python3
"""ポニーテールを多段振り子用に縦分割する。

VIEWER_QUALITY_PLAN.md §2-4「ポニーテール物理」。単一の剛体スプライトを
根元→毛先の複数バンドに分割し、それぞれ独立回転できるようにする。

境界の切れ目対策(build_face_layer.pyのhead/ponytail切り出しと同じ理屈):
各バンドは隣のバンドと数十pxオーバーラップして切り出し、重なり側の
アルファをフェザー(なだらかに0へ)する。バンド間の回転角の差は小さい
(髪のしなりなので隣接バンドはほぼ同じ角度)ため、オーバーラップ部分が
下のバンドの絵を覆い続け、境界が視認できなくなる。

**注意(2026-07-20の教訓その1)**: 上記の「アルファの重なりで隠す」仕組みは
塗りが連続した面には効くが、毛先のように細く分かれたスカスカな束では
隙間から下のバンドが透けるため、隣接バンドの角度差(=見た目の毛流れの
「キンク」)が数度でもあると継ぎ目として視認されてしまう。対策は
バンド数を増やして1関節あたりの角度差を小さくすること(4→6分割に変更)。

**注意(2026-07-20の教訓その2、本命の原因)**: ponytail.pngの根元付近には
三日月型の飾り(髪留めのリング状装飾、master座標y≈65〜230)がある。この
一体の装飾を貫通する位置(y=100)にband0/band1の境界を置いたため、
装飾の右側(band0=頭と同じ剛体transform)と左側(band1=独立回転)が
別々に動き、装飾が真っ二つに割れて見える不具合が発生した(ユーザー
指摘: 「髪の毛にある下側黒色パーツの左側」が切れる)。**教訓: バンド
境界は、意味のある一体の造形(リング・房の分かれ目など)を貫通しない
位置に置くこと。** 対策として装飾全体がband0(頭固定)に収まるよう
Y_CUTS[1]を100→210に拡張した。
"""
import json
import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "layers_zero", "ponytail.png")
OUT = os.path.join(HERE, "layers_zero", "ponytail_bands")
os.makedirs(OUT, exist_ok=True)

OVERLAP = 36  # 隣接バンドとの重なり幅(px)。フェザー幅もこれに合わせる

# 根元(頭に直結、渦巻き部分)→毛先の6バンド境界(y座標、ponytail.png内)。
# band0は三日月飾り(y≈32〜197、頭固定にしたい)を丸ごと含むよう0〜210に
# 拡張。残り(210〜567、プレーンな毛束)を5分割して曲げを分散させる
Y_CUTS = [0, 210, 280, 350, 420, 490, 567]


def centroid_x(alpha, y, halfwin=8):
    h = alpha.shape[0]
    y0, y1 = max(0, y - halfwin), min(h, y + halfwin)
    ys, xs = np.where(alpha[y0:y1] > 10)
    if len(xs) == 0:
        return None
    return float(xs.mean())


def main():
    src = Image.open(SRC).convert("RGBA")
    arr = np.array(src)
    W, H = src.size
    alpha = arr[:, :, 3]

    bands = []
    n = len(Y_CUTS) - 1
    for i in range(n):
        y_top, y_bot = Y_CUTS[i], Y_CUTS[i + 1]
        # 先頭バンドは上オーバーラップ無し(原画の縁がそのまま上端)。
        # それ以外は前バンドの領域にOVERLAP分食い込んで切り出す
        crop_y0 = y_top if i == 0 else max(0, y_top - OVERLAP)
        crop_y1 = y_bot
        band = arr[crop_y0:crop_y1].copy()
        band_alpha = band[:, :, 3].astype(np.float32)

        if i > 0:
            # 上端からOVERLAP pxをなだらかにフェードイン(0→元のアルファ)
            fade_h = min(OVERLAP, band.shape[0])
            ramp = np.linspace(0, 1, fade_h, dtype=np.float32)[:, None]
            band_alpha[:fade_h] *= ramp
        band[:, :, 3] = np.clip(band_alpha, 0, 255).astype(np.uint8)

        out_path = os.path.join(OUT, f"band{i}.png")
        Image.fromarray(band).save(out_path)

        # ピボット(親バンドへの接続点)= このバンドの上端(オーバーラップ含む
        # クロップ前のy_top)における束の中心線(アルファ重心)。バンドの
        # ローカル座標系(クロップ原点基準)に変換して記録
        pivot_x = centroid_x(alpha, y_top)
        if pivot_x is None:
            pivot_x = W * 0.5
        pivot_local = {"x": round(pivot_x, 1), "y": y_top - crop_y0}
        bands.append({
            "file": f"band{i}.png",
            "crop_y0": int(crop_y0),
            "w": W,
            "h": int(crop_y1 - crop_y0),
            "pivot_local": pivot_local,  # 親(頭 or 前バンド)に繋ぐ接続点
        })
        print(f"band{i}: y[{crop_y0}:{crop_y1}] pivot_local={pivot_local}")

    manifest = {"overlap": OVERLAP, "bands": bands}
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)

    # QC: 全バンドを元位置に重ね描きして原画と一致するか確認
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for b in bands:
        im = Image.open(os.path.join(OUT, b["file"])).convert("RGBA")
        canvas.alpha_composite(im, (0, b["crop_y0"]))
    bg = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    Image.alpha_composite(bg, canvas).convert("RGB").save(os.path.join(OUT, "_qc_recomposed.png"))
    print("done")


if __name__ == "__main__":
    main()
