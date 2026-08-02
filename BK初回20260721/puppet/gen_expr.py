#!/usr/bin/env python3
"""表情差分の生成(顔内側フェザー領域の丸ごと差し替え方式)。

PUPPET_ENGINE_DESIGN.md §A: 表情は顔の内側領域(固定フェザーマスク)を
まるごと1枚で差し替える。目・口を独立パッチにしない。
親フレームはcharacters/zero/master_bust.png(neutral採用済み)。
本スクリプトはneutralからjoy(笑顔)のimg2img候補をバッチ生成する。
"""
import os
import sys
import time
import numpy as np
from PIL import Image
from optimum.intel import OVStableDiffusionImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "..", "experiments", "ondevice_gen_test")
SRC = os.path.join(HERE, "..", "characters", "zero", "master_bust.png")
OUT = os.path.join(HERE, "..", "characters", "zero", "expr_candidates")
os.makedirs(OUT, exist_ok=True)

# 顔内側の矩形(目・眉・口を含む範囲)。フルフレームでimg2imgし、
# 後工程で境界をフェザーブレンドして合成する。
FACE_BOX = (80, 20, 432, 340)
FW, FH = 512, 512

BASE_PROMPT = ("masterpiece, best quality, ultra cute anime girl face, "
               "silver white hair with green streak highlights, "
               "cyan sparkling eyes, facing viewer, front view, "
               "kawaii, moe, clean lineart, cel shading, simple white background")
NEG = ("lowres, bad anatomy, blurry, watermark, text, realistic, 3d, "
       "hands, fingers, deformed, extra pupils, side view")

EXPRESSIONS = {
    "joy": ("big happy smile, open smiling mouth showing teeth, joyful expression, "
            "eyes closed with joy, blush", 0.5),
    "surprised": ("surprised expression, wide open eyes, small open mouth, "
                  "raised eyebrows, blush", 0.5),
    "sad": ("sad expression, downturned mouth, teary eyes, worried eyebrows", 0.45),
    "angry": ("angry expression, furrowed eyebrows, frown, pouting mouth", 0.5),
    "relaxed": ("relaxed gentle smile, half-closed calm eyes, soft expression", 0.4),
}

SEEDS = range(1, 5)


def main():
    names = sys.argv[1:] or list(EXPRESSIONS.keys())
    src_full = Image.open(SRC).convert("RGB")
    face = src_full.crop(FACE_BOX).resize((FW, FH), Image.LANCZOS)
    face.save(os.path.join(OUT, "_face_init.png"))

    pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(
        os.path.join(GEN, "meinamix_v11_ov"), safety_checker=None,
        ov_config={"CACHE_DIR": os.path.join(GEN, "ov_cache_v11")})
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True)

    for name in names:
        detail, strength = EXPRESSIONS[name]
        prompt = f"{BASE_PROMPT}, {detail}"
        for seed in SEEDS:
            out = os.path.join(OUT, f"{name}_st{int(strength*100)}_s{seed}.png")
            if os.path.exists(out):
                continue
            t0 = time.time()
            img = pipe(prompt=prompt, negative_prompt=NEG, image=face,
                       strength=strength, num_inference_steps=30, guidance_scale=7.0,
                       generator=np.random.RandomState(seed)).images[0]
            img.save(out)
            print(f"{name} seed={seed}: {time.time()-t0:.1f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
