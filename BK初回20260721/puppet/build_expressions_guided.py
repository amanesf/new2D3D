#!/usr/bin/env python3
"""閉じ目・笑い目を「Claude製下絵+低ノイズimg2img」で生成する。

素のimg2imgでは構図保持が強く目が閉じないため(ON_DEVICE_GENERATION_PLAN§3)、
先に下絵(肌塗り+まつげ弧)をOpenCVで描き込み、低strengthで画風に馴染ませる。
"""
import os
import time
import numpy as np
import cv2
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
SCALE = GEN_SIZE / (FACE[2] - FACE[0])
EYE_L = (460, 150, 516, 200)
EYE_R = (516, 146, 576, 200)


def to_gen(r):
    return (int((r[0] - FACE[0]) * SCALE), int((r[1] - FACE[1]) * SCALE),
            int((r[2] - FACE[0]) * SCALE), int((r[3] - FACE[1]) * SCALE))


base = Image.open(SRC).convert("RGB")
face = np.array(base.crop(FACE).resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS))[:, :, ::-1].copy()  # BGR

# 肌色: 頬の固定点(元1024座標)からサンプル
SKIN_PTS = {"eye_l": (474, 213), "eye_r": (562, 210)}
def skin_color(r):
    pt = SKIN_PTS["eye_l" if r == EYE_L else "eye_r"]
    gx, gy = int((pt[0] - FACE[0]) * SCALE), int((pt[1] - FACE[1]) * SCALE)
    return np.median(face[gy - 3:gy + 4, gx - 3:gx + 4].reshape(-1, 3), axis=0)


# まつげ色: 目箱内の暗部
def lash_color(r):
    g = to_gen(r)
    box = face[g[1]:g[3], g[0]:g[2]].reshape(-1, 3)
    dark = box[box.sum(1).argsort()[:box.shape[0] // 10]]
    return np.median(dark, axis=0)


def paint_closed(img, curve):  # curve: +1=⌣(blink) -1=∩(joy)
    out = img.copy()
    for r in (EYE_L, EYE_R):
        g = to_gen(r)
        skin = skin_color(r)
        lash = lash_color(r)
        # 目領域を楕円+強フェザーで肌にブレンド(矩形の硬い縁を作らない)
        cx, cy = (g[0] + g[2]) // 2, (g[1] + g[3]) // 2
        rx, ry = (g[2] - g[0]) // 2 + 8, (g[3] - g[1]) // 2 + 6
        m = np.zeros(out.shape[:2], np.float32)
        cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 9)[:, :, None]
        skin_img = np.full_like(out, skin)
        out = (skin_img * m + out * (1 - m)).astype(np.uint8)
        # まつげの弧
        cy2 = cy + 8
        w = (g[2] - g[0]) // 2 - 8
        pts = []
        for i in range(17):
            x = -w + i * (2 * w) / 16
            y = curve * (1 - (x / w) ** 2) * 16
            pts.append((int(cx + x), int(cy2 + y)))
        cv2.polylines(out, [np.array(pts, np.int32)], False, lash.tolist(), 6, cv2.LINE_AA)
    return out


guides = {
    "blink": paint_closed(face, +1),
    "joy": paint_closed(face, -1),
}
for n, g in guides.items():
    cv2.imwrite(os.path.join(RAW, f"_guide_{n}.png"), g)

COMMON = ("masterpiece, best quality, 1girl, anime style, kawaii, close-up face, "
          "silver white hair, blue eyes, cat ear headphones, closed eyes, "
          "clean lineart, cel shading, white background")
NEG = "lowres, bad anatomy, blurry, watermark, realistic, open eyes, deformed, text"
PROMPTS = {
    "blink": "calm relaxed face, eyes gently closed, smooth eyelids, small smile",
    "joy": "laughing happily, smiling closed eyes curved like arcs, cheerful",
}

pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(
    os.path.join(GEN, "meinamix_v11_ov"), ov_config={"CACHE_DIR": os.path.join(GEN, "ov_cache_v11")})
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

for name, g in guides.items():
    src = Image.fromarray(g[:, :, ::-1])
    for seed in (1, 2, 3):
        t0 = time.time()
        img = pipe(
            prompt=COMMON + ", " + PROMPTS[name],
            negative_prompt=NEG,
            image=src,
            strength=0.45,
            num_inference_steps=28,
            guidance_scale=7.0,
            generator=np.random.RandomState(seed),
        ).images[0]
        img.save(os.path.join(RAW, f"{name}g_s{seed}.png"))
        print(f"{name} seed={seed}: {time.time()-t0:.1f}s", flush=True)
print("done")
