"""IP-Adapter Plus + ControlNet + MeinaMix 側面90度、v3。

v1: 基本IP-Adapter(ip-adapter_sd15.bin、画像全体を1個のベクトルに
圧縮するグローバル埋め込み)のみ、テキストに識別要素を書かない
→ 瞳色・髪色・靴がズレた。
v2: v1のIP-Adapterのままテキストに識別要素(髪色・瞳・衣装配色)を
明示 → 大幅改善したが、テキストへの依存が残った。

v3ではユーザーの要望「テキストに頼りたくない」を受け、**IP-Adapterを
Plus版(ip-adapter-plus_sd15.safetensors、パッチ単位の特徴量を使う
ため単一ベクトルより細部の再現力が高いとされる)に差し替え、
v1同様テキストには識別要素を書かず**、画像だけでどこまで再現できるか
を検証する。
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
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.8)
    print(f"pipeline load: {time.time()-t0:.1f}s", flush=True)

    ip_image = Image.open(MASTER).convert("RGB")

    # v1同様、識別要素(色等)はテキストに書かない。画像(IP-Adapter Plus)
    # だけでどこまで再現できるかを見るのが本スパイクの目的。
    prompt = ("masterpiece, best quality, anime girl, 1girl, solo, side view, "
              "standing, simple white background")
    neg = ("lowres, bad anatomy, blurry, watermark, text, realistic, 3d, "
           "extra fingers, missing fingers, deformed, cropped, cut off, "
           "multiple views, turnaround, colored background, scenery")

    t0 = time.time()
    result = pipe(
        prompt=prompt, negative_prompt=neg, image=pose_side,
        ip_adapter_image=ip_image,
        num_inference_steps=20, guidance_scale=7.0,
        generator=torch.Generator().manual_seed(1),
    ).images[0]
    dt = time.time() - t0
    out_path = os.path.join(OUT_DIR, "calib_side90_ipa_s1_v3plus.png")
    result.save(out_path)
    print(f"generate: {dt:.1f}s -> {out_path}", flush=True)

    sim = identity_histogram_similarity(MASTER, result)
    print(f"identity histogram similarity (参考値、信頼性低いと判明済み): {sim:.3f}", flush=True)


if __name__ == "__main__":
    main()
