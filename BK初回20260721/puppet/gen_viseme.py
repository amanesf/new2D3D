#!/usr/bin/env python3
"""viseme(あいうえお5形状)の生成。

VIEWER_QUALITY_PLAN.md §1 C 2番目の項目。当初は全顔img2imgで作っていたが、
毎回顔全体(髪・陰影含む)が微妙に作り直されてしまい、ビューアで話す
ループを回すと口以外までちらついて見える問題があった
(gen_blink.pyと同根の問題)。gen_blink.pyで検証済みの
「口だけをinpaintし、マスク外は元画像に強制的に固定する」方式に統一する。
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
OUT = os.path.join(HERE, "..", "characters", "zero", "viseme_candidates")
os.makedirs(OUT, exist_ok=True)

# neutral.png(512x512)内の口を含む矩形。
# 旧値(188,300,288,352)は実際の口位置より左下(顎〜首元)にずれており、
# inpaintが口ではなく顎のあたりに口の形を描いてしまっていた(採用画像の
# 口が浮いたような不自然な形になっていた不具合の原因)。目視グリッドで
# 実際の口位置(中心付近x≈300,y≈318)に合わせて測り直した値に修正。
# 最初の修正(250,293,355,358)は口の位置自体は正しくなったが、右端が
# 頬にかかる毛束にかぶっており、inpaintがその毛束を紐/コードのような
# 形に巻き込んで口から伸びるノイズになっていた。右端を狭めて毛束を
# マスク外に追い出す
MOUTH_BOX = (248, 296, 318, 352)

BASE_PROMPT = ("masterpiece, best quality, close-up of anime girl mouth, "
               "kawaii, clean lineart, cel shading")
NEG = ("lowres, bad anatomy, blurry, watermark, text, realistic, 3d, deformed, "
       "cable, cord, string, wire, ribbon, ponytail, hair strand, tongue piercing, "
       "straw, drinking straw, tube, feeding tube, IV drip, antenna, whisker, "
       "thread, drool string, saliva string, object near mouth, isolated mouth only")

# viseme 5形状(あいうえおの口の開き方)
VISEMES = {
    "aa": ("mouth wide open, saying ah, open mouth vowel a shape, teeth visible", 0.7),
    "ih": ("mouth open, small gap between teeth, saying ee/i sound, wide smile", 0.7),
    "ou": ("small round open mouth like whistling, pursed puckered lips, saying oo", 0.7),
    "ee": ("mouth open, wide horizontal grin, teeth showing, saying ee", 0.7),
    "oh": ("round open mouth like an O shape, saying oh", 0.7),
}

SEEDS = range(1, 9)


def make_mask(size, box, feather=10):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle(box, fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather))


def main():
    names = sys.argv[1:] or list(VISEMES.keys())
    full = Image.open(SRC).convert("RGB")
    mask = make_mask(full.size, MOUTH_BOX)

    pipe = OVStableDiffusionInpaintPipeline.from_pretrained(
        os.path.join(GEN, "meinamix_v11_ov"), safety_checker=None,
        ov_config={"CACHE_DIR": os.path.join(GEN, "ov_cache_v11")})
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True)

    for name in names:
        detail, strength = VISEMES[name]
        prompt = f"{BASE_PROMPT}, {detail}"
        for seed in SEEDS:
            out = os.path.join(OUT, f"{name}_st{int(strength*100)}_s{seed}.png")
            if os.path.exists(out):
                continue
            t0 = time.time()
            gen = pipe(prompt=prompt, negative_prompt=NEG, image=full, mask_image=mask,
                       strength=strength, num_inference_steps=30, guidance_scale=7.0,
                       generator=np.random.RandomState(seed)).images[0]
            if gen.size != full.size:
                gen = gen.resize(full.size, Image.LANCZOS)
            composed = Image.composite(gen, full, mask)
            composed.save(out)
            print(f"{name} seed={seed}: {time.time()-t0:.1f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
