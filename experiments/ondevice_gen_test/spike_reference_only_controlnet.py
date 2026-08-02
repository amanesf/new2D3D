"""Reference-only ControlNet + MeinaMix 側面90度、v4。

IP-Adapter系(v1基本/v3 Plus)は画像を1個or少数のベクトル/パッチ埋め込みに
圧縮してからUNetのクロスアテンションに渡す方式で、どちらもテキスト無しでは
識別要素(特にv3では猫耳ヘッドセット)を安定して保持できなかった。

Reference-onlyは埋め込みへの圧縮を経由せず、参照画像をVAEでエンコードした
latentをUNetに一緒に流し、**各デノイズステップの自己注意(self-attention)に
参照画像自身のキー/バリューを直接混ぜる**(+グループ正規化統計も合わせる)、
sd-webui-controlnet由来の方式。圧縮のボトルネックが無い分、識別要素の
再現力が高いと期待される、テキストに頼らない路線で最後に試す候補。
"""
import os
import time

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, DiffusionPipeline, UniPCMultistepScheduler
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
    ref_image = Image.open(MASTER).convert("RGB")

    t0 = time.time()
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32)
    pipe = DiffusionPipeline.from_pretrained(
        "Meina/MeinaMix_V11", controlnet=controlnet,
        custom_pipeline="stable_diffusion_controlnet_reference",
        torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    print(f"pipeline load: {time.time()-t0:.1f}s", flush=True)

    # v1/v3と同じく識別要素はテキストに書かない(画像だけでの再現度を見る)
    prompt = ("masterpiece, best quality, anime girl, 1girl, solo, side view, "
              "standing, simple white background")
    neg = ("lowres, bad anatomy, blurry, watermark, text, realistic, 3d, "
           "extra fingers, missing fingers, deformed, cropped, cut off, "
           "multiple views, turnaround, colored background, scenery")

    t0 = time.time()
    result = pipe(
        prompt=prompt, negative_prompt=neg, image=pose_side,
        ref_image=ref_image,
        reference_attn=True, reference_adain=True,
        num_inference_steps=20, guidance_scale=7.0,
        generator=torch.Generator().manual_seed(1),
    ).images[0]
    dt = time.time() - t0
    out_path = os.path.join(OUT_DIR, "calib_side90_refonly_s1.png")
    result.save(out_path)
    print(f"generate: {dt:.1f}s -> {out_path}", flush=True)

    sim = identity_histogram_similarity(MASTER, result)
    print(f"identity histogram similarity (参考値、信頼性低いと判明済み): {sim:.3f}", flush=True)


if __name__ == "__main__":
    main()
