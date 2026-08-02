"""img2imgチェーン(0度→45度)、全身プロンプト漏れの修正+
img2img側は識別要素を書かない版。

前回(⑬)は"full body, standing"を書き漏らしバストアップ構図に
なっていた。0度(txt2img)には書き足す。また、img2imgは既存画像
(実ピクセル)から始めるので、識別要素(髪色・衣装等)を毎回書き直す
必要は無い(ユーザー指摘)。45度側は構図・回転の指示だけに絞り、
77トークンの余裕を無駄にしない。
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

PROMPT_0 = ("best quality, anime girl, 1girl, solo, full body, standing, "
            "front view, silver white hair with pink and green streak "
            "highlights, ponytail, green eyes, black cat ear headset with "
            "mint green glow, white crop top, black shorts, puffed "
            "sleeves, white sheer cape with teal trim, white thigh-high "
            "boots with teal accents, simple white background")
PROMPT_45 = "best quality, full body, standing, three-quarter view, simple white background"
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
        prompt=PROMPT_0, negative_prompt=NEG,
        height=H, width=W, num_inference_steps=40, guidance_scale=7.0,
        generator=torch.Generator().manual_seed(SEED),
    ).images[0]
    img_0.save(os.path.join(OUT_DIR, "calib_chain_fullbody_0deg.png"))
    print(f"0deg (txt2img): {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    img_45 = img2img(
        prompt=PROMPT_45, negative_prompt=NEG,
        image=img_0, strength=STRENGTH,
        num_inference_steps=40, guidance_scale=7.0,
        generator=torch.Generator().manual_seed(SEED),
    ).images[0]
    img_45.save(os.path.join(OUT_DIR, "calib_chain_fullbody_45deg.png"))
    print(f"45deg (img2img from 0deg): {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
