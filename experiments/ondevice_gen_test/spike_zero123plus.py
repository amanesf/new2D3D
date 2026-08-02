"""Zero123++のCPU実行可否スパイク(最小構成)。

diffusersのcustom_pipeline経由でロードできるか・CPU(float32)で
1枚(実際は6ビュー同時出力)動くかだけをまず確認する。
"""
import os
import time

import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
MASTER = os.path.join(ROOT, "characters", "zero", "master_v2_fullbody.png")
OUT_DIR = os.path.join(HERE, "spike_zero123plus")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    t0 = time.time()
    pipe = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.1",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    print(f"pipeline load: {time.time()-t0:.1f}s", flush=True)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to("cpu")

    im = Image.open(MASTER).convert("RGB")
    # 正方形にパディング(Zero123++は正方形入力を想定)
    side = max(im.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    canvas = canvas.resize((320, 320))
    canvas.save(os.path.join(OUT_DIR, "input_320.png"))

    t0 = time.time()
    result = pipe(canvas, num_inference_steps=20).images[0]
    dt = time.time() - t0
    out_path = os.path.join(OUT_DIR, "views_grid.png")
    result.save(out_path)
    print(f"generate: {dt:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
