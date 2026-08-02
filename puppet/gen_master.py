#!/usr/bin/env python3
"""ゼロベース新マスター生成(面影維持版)。

原画のバストアップクロップからimg2imgで再構成する:
- 面影(高ポニーテール・緑メッシュ銀髪・シアン目・黒x ミント猫耳ヘッドセット)は
  init画像とプロンプト両方で固定
- リグ都合の変更点だけ生成に任せる: マイクアーム除去・口を閉じる・
  顔に何もかからない・可愛さ(目大きめ・blush)増し
- strengthを2水準振る(低=面影寄り、高=再構成寄り)
"""
import os
import time
import numpy as np
from PIL import Image
from optimum.intel import OVStableDiffusionImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "..", "experiments", "ondevice_gen_test")
SRC = os.path.join(HERE, "..", "characters", "silver_ponytail", "full.jpg")
OUT = os.path.join(HERE, "..", "characters", "zero", "candidates")
os.makedirs(OUT, exist_ok=True)

# バストアップクロップ(頭〜胸上、ポニーテール含む)→ 512x704
W, H = 512, 704
src = Image.open(SRC).convert("RGB").crop((280, 10, 760, 670)).resize((W, H), Image.LANCZOS)
src.save(os.path.join(OUT, "_init.png"))

PROMPT = ("masterpiece, best quality, ultra cute anime girl, 1girl, solo, "
          "bust portrait, facing viewer, front view, "
          "huge expressive sparkling cyan eyes, small closed smiling mouth, "
          "silver white hair with green streak highlights, high ponytail, "
          "black cat ear headphones with mint green glow, white futuristic crop top, "
          "kawaii, moe, blush, clean lineart, cel shading, "
          "arms down at sides, nothing covering face, simple white background")
NEG = ("lowres, bad anatomy, blurry, watermark, text, realistic, 3d, "
       "open mouth, hands, fingers, microphone, boom mic, closed eyes, "
       "side view, extra pupils, deformed, twintails")

STRENGTHS = (0.45, 0.6)
SEEDS = range(1, 9)

pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(
    os.path.join(GEN, "meinamix_v11_ov"), safety_checker=None,
    ov_config={"CACHE_DIR": os.path.join(GEN, "ov_cache_v11")})
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True)

for st in STRENGTHS:
    for seed in SEEDS:
        out = os.path.join(OUT, f"bust_st{int(st*100)}_s{seed}.png")
        if os.path.exists(out):
            continue
        t0 = time.time()
        img = pipe(prompt=PROMPT, negative_prompt=NEG, image=src,
                   strength=st, num_inference_steps=30, guidance_scale=7.0,
                   generator=np.random.RandomState(seed)).images[0]
        img.save(out)
        print(f"st={st} seed={seed}: {time.time()-t0:.1f}s", flush=True)
print("done")
