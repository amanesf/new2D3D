"""生のMeinaMix(ControlNet・IP-Adapterなし)で0度/45度/90度を、
同じseedのまま視点タグだけ差し替えて生成し、同一性・連続性を見る。

⑩(spike_raw_txt2img_90deg.py)で「テキストのみでも実際に回転する」と
分かったが、そこでは角度ごとにseedが違ったため、同一性の変化が
プロンプト(視点タグ)由来かseed由来か切り分けられなかった。本スクリプト
は同じseedに固定し、視点タグ("front view"/"three-quarter view"/
"side view, profile")だけを差し替えて3枚生成することで、同一seed内で
どこまで同一キャラとして繋がるかを見る。全プロンプト77トークン以内。
"""
import os
import time

import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "spike_ipadapter")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 3  # ⑩でユーザーが猫耳を確認できた実績のあるseed

VIEWS = [
    ("0deg", "front view"),
    ("45deg", "three-quarter view"),
    ("90deg", "side view, profile"),
]

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
    pipe = StableDiffusionPipeline.from_pretrained(
        "Meina/MeinaMix_V11", torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    print(f"pipeline load: {time.time()-t0:.1f}s", flush=True)

    for label, view in VIEWS:
        prompt = BASE.format(view=view)
        t0 = time.time()
        result = pipe(
            prompt=prompt, negative_prompt=NEG,
            height=704, width=512,
            num_inference_steps=40, guidance_scale=7.0,
            generator=torch.Generator().manual_seed(SEED),
        ).images[0]
        dt = time.time() - t0
        out_path = os.path.join(OUT_DIR, f"calib_raw_angles_seed{SEED}_{label}.png")
        result.save(out_path)
        print(f"{label} generate: {dt:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
