"""img2imgチェーンによる連続性改善の実験(低解像度、速度優先)。

⑪までの検証は角度ごとに独立生成(ノイズから毎回ゼロ生成)していたため、
同一seedでもヘッドセット形状等が微妙にズレていた。本スクリプトは
0度をtxt2imgで作った後、**45度は0度の画像からimg2img、90度は45度の
画像からimg2img**という連続チェーンにして、実ピクセルを角度間で
引き継ぐことで同一性が安定するかを見る。

解像度は256x352(面積1/4、計画書§6のミニマル構成に準拠)に落として
高速に反復できるようにする。strengthは0.6(回転できる自由度を残しつつ
前段のピクセルもある程度引き継ぐ、中間的な値)で試す。
"""
import os
import time

import torch
from diffusers import (StableDiffusionImg2ImgPipeline, StableDiffusionPipeline,
                        UniPCMultistepScheduler)
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "spike_ipadapter")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 3
W, H = 256, 352  # 512x704の面積1/4
STRENGTH = 0.6

BASE = ("best quality, anime girl, 1girl, solo, {view}, "
        "silver white hair with pink and green streak highlights, "
        "ponytail, green eyes, black cat ear headset with mint green "
        "glow, white crop top, black shorts, white puffed sleeves, "
        "fingerless gloves, white sheer cape with teal trim, white "
        "thigh-high boots with teal accents, simple white background")
NEG = ("lowres, bad anatomy, blurry, watermark, text, realistic, 3d, "
       "extra fingers, missing fingers, deformed, cropped, cut off, "
       "multiple views, turnaround, colored background, scenery")


def main():
    t0 = time.time()
    txt2img = StableDiffusionPipeline.from_pretrained(
        "Meina/MeinaMix_V11", torch_dtype=torch.float32, safety_checker=None)
    txt2img.scheduler = UniPCMultistepScheduler.from_config(txt2img.scheduler.config)
    img2img = StableDiffusionImg2ImgPipeline(**txt2img.components)
    print(f"pipeline load: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    img_0 = txt2img(
        prompt=BASE.format(view="front view"), negative_prompt=NEG,
        height=H, width=W, num_inference_steps=40, guidance_scale=7.0,
        generator=torch.Generator().manual_seed(SEED),
    ).images[0]
    img_0.save(os.path.join(OUT_DIR, "calib_chain_lowres_0deg.png"))
    print(f"0deg (txt2img): {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    img_45 = img2img(
        prompt=BASE.format(view="three-quarter view"), negative_prompt=NEG,
        image=img_0, strength=STRENGTH,
        num_inference_steps=40, guidance_scale=7.0,
        generator=torch.Generator().manual_seed(SEED),
    ).images[0]
    img_45.save(os.path.join(OUT_DIR, "calib_chain_lowres_45deg.png"))
    print(f"45deg (img2img from 0deg): {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    img_90 = img2img(
        prompt=BASE.format(view="side view, profile"), negative_prompt=NEG,
        image=img_45, strength=STRENGTH,
        num_inference_steps=40, guidance_scale=7.0,
        generator=torch.Generator().manual_seed(SEED),
    ).images[0]
    img_90.save(os.path.join(OUT_DIR, "calib_chain_lowres_90deg.png"))
    print(f"90deg (img2img from 45deg): {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
