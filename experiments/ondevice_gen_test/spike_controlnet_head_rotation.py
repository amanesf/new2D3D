"""頭部キーフレーム回転のControlNet実現性スパイク(最小構成、校正1枚)。

optimum-intelのOVStableDiffusionControlNetPipelineは現行版(1.27.0)に
存在しないため廃止と判明。素のdiffusers(PyTorch CPU)でControlNet+
OpenPoseが動くか、90度(側面)1枚だけでまず確認する。
"""
import os
import sys
import time

import numpy as np
import torch
from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
MASTER = os.path.join(ROOT, "characters", "zero", "master_v2_fullbody.png")
OUT_DIR = os.path.join(HERE, "spike_controlnet")
os.makedirs(OUT_DIR, exist_ok=True)


def get_openpose_guide():
    """マスターからOpenPose骨格を抽出(校正1枚のみなので毎回検出でよい)。"""
    detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    master = Image.open(MASTER).convert("RGB")
    pose = detector(master, hand_and_face=False)
    pose.save(os.path.join(OUT_DIR, "pose_front_detected.png"))
    return pose


def make_side_view_guide(pose_front: Image.Image) -> Image.Image:
    """90度(側面)用の簡易2D変換骨格。3Dプロキシメッシュは組まず、
    横幅を圧縮するだけの決定的変換に留める(調整コストを抑える方針)。
    """
    w, h = pose_front.size
    squashed = pose_front.resize((max(1, int(w * 0.55)), h))
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(squashed, ((w - squashed.width) // 2, 0))
    canvas.save(os.path.join(OUT_DIR, "pose_side90_guide.png"))
    return canvas


def main():
    t0 = time.time()
    pose_front = get_openpose_guide()
    print(f"openpose detect: {time.time()-t0:.1f}s", flush=True)

    pose_side = make_side_view_guide(pose_front)

    t0 = time.time()
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32)
    print(f"controlnet load: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "Meina/MeinaMix_V11", controlnet=controlnet,
        torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    print(f"sd pipeline load: {time.time()-t0:.1f}s", flush=True)

    prompt = ("masterpiece, best quality, anime girl, 1girl, solo, side view, "
               "silver white hair with green streak highlights, high ponytail, "
               "cyan eyes, black cat ear headphones with mint green glow, "
               "white futuristic jacket, short jacket, shorts, thigh high boots, "
               "kawaii, clean lineart, cel shading, simple white background")
    neg = ("lowres, bad anatomy, blurry, watermark, text, realistic, 3d, "
           "extra fingers, missing fingers, deformed, cropped, cut off, "
           "multiple views, turnaround, colored background, scenery")

    t0 = time.time()
    result = pipe(
        prompt=prompt, negative_prompt=neg, image=pose_side,
        num_inference_steps=20, guidance_scale=7.0,
        generator=torch.Generator().manual_seed(1),
    ).images[0]
    dt = time.time() - t0
    out_path = os.path.join(OUT_DIR, "calib_side90_s1.png")
    result.save(out_path)
    print(f"generate: {dt:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
