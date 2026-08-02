#!/usr/bin/env python3
"""v2生産ライン: 顔まるごと表情候補のバッチ生成(第2次)。

第1次の知見を反映:
- strength 0.55/0.65では表情が変わらない → 表情系は0.70/0.85に引き上げ
- 構図が引きのバストショットにドリフト → プロンプトで顔クローズアップを固定し、
  残るズレは登録時のテンプレートマッチング位置合わせで吸収
- セーフティチェッカー誤検知の黒画像 → 無効化
- 保存は512px JPEG q90(1枚50〜80KB。位置合わせ後の再拡大に耐える解像度を確保)

usage: python3 gen_face_bank.py [target ...]
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
RAW = os.path.join(HERE, "expressions", "raw")
os.makedirs(RAW, exist_ok=True)

face = Image.open(os.path.join(HERE, "face", "_source_512.png")).convert("RGB")

COMMON = ("masterpiece, best quality, 1girl, anime style, kawaii, "
          "close-up portrait, face focus, head fills the frame, "
          "silver white hair, green streaks, blue eyes, cat ear headphones, "
          "clean lineart, cel shading, white background")
NEG = ("lowres, bad anatomy, blurry, watermark, realistic, extra pupils, "
       "asymmetric eyes, deformed, text, full body, upper body, wide shot, "
       "small head, chest, shoulders")

# name: (prompt, extra_negative, strengths)
TARGETS = {
    "blink": ("calm relaxed face, eyes fully closed, gentle closed eyelids with lashes, "
              "small closed smile", ", open eyes, half-open eyes", (0.70, 0.85)),
    "joy": ("laughing joyfully, closed smiling eyes curved upward like arches, "
            "wide open happy smile", ", open eyes", (0.70, 0.85)),
    "surprised": ("surprised face, wide open eyes, small round open mouth, "
                  "raised eyebrows", "", (0.60, 0.75)),
    "mouth_aa": ("talking, mouth wide open saying ah, open eyes looking forward",
                 ", closed mouth", (0.60, 0.75)),
}
SEEDS = range(1, 9)

targets = sys.argv[1:] or list(TARGETS)
pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(
    os.path.join(GEN, "meinamix_v11_ov"), safety_checker=None,
    ov_config={"CACHE_DIR": os.path.join(GEN, "ov_cache_v11")})
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True)

for name in targets:
    prompt, extra_neg, strengths = TARGETS[name]
    for st in strengths:
        for seed in SEEDS:
            out = os.path.join(RAW, f"{name}_c{int(st*100)}_s{seed}.jpg")
            if os.path.exists(out):
                continue
            t0 = time.time()
            img = pipe(prompt=COMMON + ", " + prompt, negative_prompt=NEG + extra_neg,
                       image=face, strength=st, num_inference_steps=25,
                       guidance_scale=7.0,
                       generator=np.random.RandomState(seed)).images[0]
            img.save(out, quality=90)
            print(f"{name} st={st} seed={seed}: {time.time()-t0:.1f}s", flush=True)
print("done")
