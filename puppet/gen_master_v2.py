#!/usr/bin/env python3
"""zeroキャラ マスター再生成(v2、ポーズ再設計版)。

SUPER_LIVE2D_V3_PLAN.md のL0(生成をリグに従わせる)を受けた再生成。
Spike S1でponytailが腕・グローブに重なる構図(旧master_bust.png)は
inpaint不可能と判明したため、ユーザー提供のポーズ参照画像
(characters/zero/pose_ref/original_fullbody.jpeg — ポニーテールが
体・腕と一切重ならず全身が写る)を初期画像としてimg2imgで面影
(銀髪+緑メッシュ、シアン目、黒xミント猫耳ヘッドセット)を確認・
補正する。ポーズ自体は初期画像をほぼ維持するため低〜中strengthで
複数候補を出す。
"""
import os
import time
import numpy as np
from PIL import Image, ImageOps
from optimum.intel import OVStableDiffusionImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "..", "experiments", "ondevice_gen_test")
SRC = os.path.join(HERE, "..", "characters", "zero", "pose_ref", "original_fullbody.jpeg")
OUT = os.path.join(HERE, "..", "characters", "zero", "candidates_v2")
os.makedirs(OUT, exist_ok=True)

# 全身+ポニーテールが収まる作業キャンバス(512x704、既存adoptedと同一比率)。
# pose_refは800x800でキャラのbboxが(80,734)x(55,783)なのでほぼ全面を使う。
W, H = 512, 704
raw = Image.open(SRC).convert("RGB")
fitted = ImageOps.pad(raw, (W, H), color=(255, 255, 255), centering=(0.5, 0.42))
fitted.save(os.path.join(OUT, "_init.png"))

PROMPT = ("masterpiece, best quality, ultra cute anime girl, 1girl, solo, "
          "full body, standing, facing viewer, front view, "
          "huge expressive sparkling cyan eyes, small closed smiling mouth, "
          "silver white hair with green streak highlights, high ponytail flowing to the side, "
          "black cat ear headphones with mint green glow, white futuristic crop top, "
          "translucent cape panels, shorts, thigh high boots, "
          "kawaii, moe, blush, clean lineart, cel shading, "
          "arms down at sides, nothing covering face, simple white background")
NEG = ("lowres, bad anatomy, blurry, watermark, text, realistic, 3d, "
       "open mouth, extra fingers, missing fingers, microphone, boom mic, closed eyes, "
       "side view, extra pupils, deformed, twintails, cropped, cut off, "
       "arms overlapping hair, hair covering arms")

STRENGTHS = (0.35, 0.5)
SEEDS = range(1, 5)

pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(
    os.path.join(GEN, "meinamix_v11_ov"), safety_checker=None,
    ov_config={"CACHE_DIR": os.path.join(GEN, "ov_cache_v11")})
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True)

for st in STRENGTHS:
    for seed in SEEDS:
        out = os.path.join(OUT, f"master_st{int(st*100)}_s{seed}.png")
        if os.path.exists(out):
            continue
        t0 = time.time()
        img = pipe(prompt=PROMPT, negative_prompt=NEG, image=fitted,
                   strength=st, num_inference_steps=30, guidance_scale=7.0,
                   generator=np.random.RandomState(seed)).images[0]
        img.save(out)
        print(f"st={st} seed={seed}: {time.time()-t0:.1f}s", flush=True)
print("done")
