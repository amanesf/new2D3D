#!/usr/bin/env python3
"""表情差分をローカルAI(MeinaMix V11 OpenVINO)で生成する。

silver_ponytailの顔領域をクロップ→表情プロンプトでimg2img→
候補を puppet/expressions/raw/ に保存。目視QCで採用seedを決めた後、
extract_patches.py で目・口パッチだけ切り出してレイヤー登録する。
(全体再生成はしない: 採用されるのは目・口の矩形領域のみで、
顔の他の部分は常に原画ピクセルが残る)
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
RAW = os.path.join(HERE, "expressions", "raw")
os.makedirs(RAW, exist_ok=True)

# 顔の正方形クロップ(元1024座標)。目(146-200)・口(195-224)を含む
FACE = (420, 60, 660, 300)  # 240x240
GEN_SIZE = 512

base = Image.open(SRC).convert("RGB")
face = base.crop(FACE).resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS)
face.save(os.path.join(RAW, "_source_face.png"))

COMMON = ("masterpiece, best quality, 1girl, anime style, kawaii, close-up face, "
          "silver white hair, green streaks, blue eyes, cat ear headphones, "
          "clean lineart, cel shading, white background")
NEG = ("lowres, bad anatomy, blurry, watermark, realistic, extra pupils, "
       "asymmetric eyes, deformed, text")

EXPRESSIONS = {
    # name: (追加プロンプト, 追加ネガ, strength)
    "surprised": ("surprised expression, wide open eyes, small round mouth open in shock",
                  "smile, closed mouth, closed eyes", 0.6),
    "joy":       ("laughing happily, closed smiling eyes curved upward, big open smile",
                  "open eyes, sad", 0.65),
    "blink":     ("calm face, eyes fully closed, gentle closed eyelids, relaxed small smile",
                  "open eyes, wide eyes", 0.65),
    "mouth_aa":  ("talking, mouth open mid-speech, relaxed eyes",
                  "closed mouth", 0.55),
}

pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(
    os.path.join(GEN, "meinamix_v11_ov"), ov_config={"CACHE_DIR": os.path.join(GEN, "ov_cache_v11")})
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

for name, (extra, extra_neg, strength) in EXPRESSIONS.items():
    for seed in (1, 2, 3):
        t0 = time.time()
        img = pipe(
            prompt=COMMON + ", " + extra,
            negative_prompt=NEG + ", " + extra_neg,
            image=face,
            strength=strength,
            num_inference_steps=28,
            guidance_scale=7.5,
            generator=np.random.RandomState(seed),
        ).images[0]
        img.save(os.path.join(RAW, f"{name}_s{seed}.png"))
        print(f"{name} seed={seed}: {time.time()-t0:.1f}s", flush=True)
print("done")
