"""img2imgチェーン(0度→45度)をフル解像度(512x704)で再検証。

⑫(256x352)は同一性の連続性は完璧だったが、低解像度による質の劣化
(SD1.5系が学習解像度を大きく下回ると細部が崩れる現象)が別途混ざって
いたため、解像度だけを変数として切り分ける。strength等の設定は⑫と
完全に同一のまま、フル解像度(512x704)で0度→45度の1ステップのみ再実行。
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
W, H = 512, 704
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
    img_0.save(os.path.join(OUT_DIR, "calib_chain_fullres_0deg.png"))
    print(f"0deg (txt2img): {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    img_45 = img2img(
        prompt=BASE.format(view="three-quarter view"), negative_prompt=NEG,
        image=img_0, strength=STRENGTH,
        num_inference_steps=40, guidance_scale=7.0,
        generator=torch.Generator().manual_seed(SEED),
    ).images[0]
    img_45.save(os.path.join(OUT_DIR, "calib_chain_fullres_45deg.png"))
    print(f"45deg (img2img from 0deg): {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
