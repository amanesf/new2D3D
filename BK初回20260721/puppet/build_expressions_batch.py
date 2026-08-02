#!/usr/bin/env python3
"""閉じ目・笑い目の大量バッチ生成(数打って目視で選ぶ方式)。

下絵ガイド方式は品質不足だったため転換: strength 2水準 x seed 8個を
素のimg2imgで量産し、目・口パッチとして使える個体をコンタクトシートから選ぶ。
パッチ方式では目領域しか使わないので、周辺の同一性ブレは選別で許容できる。
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

FACE = (420, 60, 660, 300)
GEN_SIZE = 512
base = Image.open(SRC).convert("RGB")
face = base.crop(FACE).resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS)

COMMON = ("masterpiece, best quality, 1girl, anime style, kawaii, close-up face, "
          "silver white hair, green streaks, blue eyes, cat ear headphones, "
          "clean lineart, cel shading, white background")
NEG = ("lowres, bad anatomy, blurry, watermark, realistic, extra pupils, "
       "asymmetric eyes, deformed, text, open eyes, half-open eyes")

TARGETS = {
    "blink": "calm relaxed face, eyes fully closed, gentle closed eyelids with lashes, small closed smile",
    "joy": "laughing joyfully, closed smiling eyes curved upward like arches, wide open happy smile",
}
STRENGTHS = (0.65, 0.75)
SEEDS = range(1, 9)

pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(
    os.path.join(GEN, "meinamix_v11_ov"), ov_config={"CACHE_DIR": os.path.join(GEN, "ov_cache_v11")})
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

for name, prompt in TARGETS.items():
    for st in STRENGTHS:
        for seed in SEEDS:
            t0 = time.time()
            img = pipe(
                prompt=COMMON + ", " + prompt,
                negative_prompt=NEG,
                image=face,
                strength=st,
                num_inference_steps=28,
                guidance_scale=7.5,
                generator=np.random.RandomState(seed),
            ).images[0]
            img.save(os.path.join(RAW, f"{name}_b{int(st*100)}_s{seed}.png"))
            print(f"{name} st={st} seed={seed}: {time.time()-t0:.1f}s", flush=True)
print("done")
