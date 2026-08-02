"""IP-Adapter+ControlNet+MeinaMix 側面90度、v5(高品質設定)。

v2(現時点の実用ベスト)をベースに、正面マスター(30ステップ)との
差を埋めるためステップ数を増やす(20→40)。また、v2ではプロンプトが
87トークンでCLIPの77トークン上限を超え末尾(背景・線画指定)が
切り捨てられていたため、優先度の低いスタイルタグを削って収める。
"""
import os
import time

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
MASTER = os.path.join(ROOT, "characters", "zero", "master_v2_fullbody.png")
OUT_DIR = os.path.join(HERE, "spike_ipadapter")
POSE_GUIDE = os.path.join(OUT_DIR, "pose_side90_guide_v2.png")


def identity_histogram_similarity(ref_path, gen_img):
    ref = cv2.imread(ref_path)
    gen = cv2.cvtColor(np.array(gen_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gen = cv2.resize(gen, (ref.shape[1], ref.shape[0]))

    def hist(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(h, h)
        return h

    return cv2.compareHist(hist(ref), hist(gen), cv2.HISTCMP_CORREL)


def main():
    pose_side = Image.open(POSE_GUIDE).convert("RGB")

    t0 = time.time()
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "Meina/MeinaMix_V11", controlnet=controlnet,
        torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(0.8)
    print(f"pipeline load: {time.time()-t0:.1f}s", flush=True)

    ip_image = Image.open(MASTER).convert("RGB")

    # 77トークン上限に収めるため、スタイル寄りのタグ(clean lineart等)は
    # 削り、識別要素(色)と背景指定を優先。
    prompt = ("best quality, anime girl, 1girl, solo, side view, "
              "silver white hair with pink and green streak highlights, "
              "high ponytail, green eyes, black cat ear headset with mint "
              "green glow, white crop top, black shorts, white sheer puffed "
              "sleeves, fingerless gloves, white sheer cape with teal "
              "trim, white thigh-high boots with teal accents, "
              "simple white background")
    neg = ("lowres, bad anatomy, blurry, watermark, text, realistic, 3d, "
           "extra fingers, missing fingers, deformed, cropped, cut off, "
           "multiple views, turnaround, colored background, scenery, tail, "
           "glowing ring, light ring, motion lines, brown eyes, red eyes, "
           "blue eyes, black hair, purple shoes")

    t0 = time.time()
    result = pipe(
        prompt=prompt, negative_prompt=neg, image=pose_side,
        ip_adapter_image=ip_image,
        num_inference_steps=40, guidance_scale=7.0,
        generator=torch.Generator().manual_seed(1),
    ).images[0]
    dt = time.time() - t0
    out_path = os.path.join(OUT_DIR, "calib_side90_ipa_s1_v5hq.png")
    result.save(out_path)
    print(f"generate: {dt:.1f}s -> {out_path}", flush=True)

    sim = identity_histogram_similarity(MASTER, result)
    print(f"identity histogram similarity (参考値、信頼性低いと判明済み): {sim:.3f}", flush=True)


if __name__ == "__main__":
    main()
