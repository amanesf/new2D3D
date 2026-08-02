"""生のMeinaMix(ControlNetもIP-Adapterも無し、テキストのみ)で
90度側面を指定できるかの再検証。

元計画(SUPER_LIVE2D_V3_PLAN.md §6)は「ControlNet等の姿勢誘導ツールが
無いため、単純img2img/txt2imgのみで角度を作る。低strengthでは構図に
引っ張られ回転しないと予想され」という**未検証の予想**のもと、
高strength img2img/txt2imgでの角度指定を計画していた。実際にはこの後
すぐControlNetの導入に進み、この「生のモデルだけで90度を指定できるか」
という前提そのものは一度も直接検証していなかった。ユーザー指摘を受け、
ここで直接検証する。

txt2img(初期画像無し、ControlNet無し、IP-Adapter無し)でプロンプトに
"side view, profile"と識別要素を書き、実際に側面を向くかを見る。
77トークン以内に収めること。
"""
import os
import time

import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "spike_ipadapter")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        "Meina/MeinaMix_V11", torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    print(f"pipeline load: {time.time()-t0:.1f}s", flush=True)

    prompt = ("best quality, anime girl, 1girl, solo, side view, profile, "
              "silver white hair with pink and green streak highlights, "
              "ponytail, green eyes, black cat ear headset with mint green "
              "glow, white crop top, black shorts, white puffed sleeves, "
              "fingerless gloves, white sheer cape with teal trim, white "
              "thigh-high boots with teal accents, simple white background")
    neg = ("lowres, bad anatomy, blurry, watermark, text, realistic, 3d, "
           "extra fingers, missing fingers, deformed, cropped, cut off, "
           "multiple views, turnaround, colored background, scenery, "
           "front view, facing viewer")

    for seed in (1, 2, 3):
        t0 = time.time()
        result = pipe(
            prompt=prompt, negative_prompt=neg,
            height=704, width=512,
            num_inference_steps=40, guidance_scale=7.0,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]
        dt = time.time() - t0
        out_path = os.path.join(OUT_DIR, f"calib_raw_txt2img_90deg_s{seed}.png")
        result.save(out_path)
        print(f"seed{seed} generate: {dt:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
