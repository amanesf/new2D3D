#!/usr/bin/env python3
"""まばたきキーフレーム(closed/half)の生成。

VIEWER_QUALITY_PLAN.md §1 C 最優先項目。

経緯: 全顔img2img(gen_viseme.pyと同じ手法)は、目を閉じさせるのに
必要なstrengthで髪色・ヘッドセット等がドリフトし、弱いstrengthでは
目が開いたままだった。目領域だけを切り出すimg2imgも試したが、
closeupクロップだと構造保持が強く働き閉眼しないか、逆に強すぎると
別人の顔が描かれてしまい板挟みだった。

**inpaint(マスク指定)パイプラインに切り替えたところ良好**: マスク外
(髪・ヘッドセット・肌・口)は完全に元画像のまま保持されつつ、マスク内
だけが周辺文脈を見ながら自然に生成されるため、閉眼指示が効きつつ
ドリフトも起きない。
"""
import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from optimum.intel import OVStableDiffusionInpaintPipeline
from diffusers import DPMSolverMultistepScheduler

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "..", "experiments", "ondevice_gen_test")
SRC = os.path.join(HERE, "..", "characters", "zero", "expr_adopted", "neutral.png")
OUT = os.path.join(HERE, "..", "characters", "zero", "blink_candidates")
os.makedirs(OUT, exist_ok=True)

# neutral.png(512x512)内の目を含む矩形。
# 旧値(110,175,360,280)は実際の目位置とずれており、左のヘッドセット
# イヤーカップまで含んでしまっていた(blink合成時にヘッドセットの色が
# わずかに変わる不具合の原因)。目視グリッドで測り直した値に修正
EYE_BOX = (185, 195, 360, 290)

BASE_PROMPT = ("masterpiece, best quality, ultra cute anime girl face, "
               "silver white hair, facing viewer, front view, kawaii, "
               "clean lineart, cel shading")
NEG = ("lowres, bad anatomy, blurry, watermark, text, realistic, 3d, deformed, "
       "extra pupils")

BLINKS = {
    "half": ("(eyes half-closed:1.3), drowsy relaxed heavy eyelids halfway down, "
             "sleepy half-lidded gentle blink", 0.75),
    "closed": ("(eyes fully closed:1.5), (shut eyelids:1.4), peaceful closed eyes, "
               "eyelashes visible, no visible iris or pupil, both eyes completely shut", 0.9),
}

SEEDS = range(1, 5)


def make_mask(size, box, feather=10):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle(box, fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather))


def main():
    names = sys.argv[1:] or list(BLINKS.keys())
    full = Image.open(SRC).convert("RGB")
    mask = make_mask(full.size, EYE_BOX)

    pipe = OVStableDiffusionInpaintPipeline.from_pretrained(
        os.path.join(GEN, "meinamix_v11_ov"), safety_checker=None,
        ov_config={"CACHE_DIR": os.path.join(GEN, "ov_cache_v11")})
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True)

    for name in names:
        detail, strength = BLINKS[name]
        prompt = f"{BASE_PROMPT}, {detail}"
        for seed in SEEDS:
            out = os.path.join(OUT, f"{name}_st{int(strength*100)}_s{seed}.png")
            if os.path.exists(out):
                continue
            t0 = time.time()
            gen = pipe(prompt=prompt, negative_prompt=NEG, image=full, mask_image=mask,
                       strength=strength, num_inference_steps=30, guidance_scale=7.0,
                       generator=np.random.RandomState(seed)).images[0]
            # inpaintパイプラインの出力はマスク外もVAE往復等でわずかに変化するため、
            # マスク外は必ず元画像そのままに戻す(これが無いとhair等が全面的に
            # わずかに再生成されてビューアでのちらつき原因になる)
            if gen.size != full.size:
                gen = gen.resize(full.size, Image.LANCZOS)
            composed = Image.composite(gen, full, mask)
            composed.save(out)
            print(f"{name} seed={seed}: {time.time()-t0:.1f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
