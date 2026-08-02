import time
import numpy as np
from optimum.intel import OVStableDiffusionImg2ImgPipeline
from diffusers.utils import load_image

init_image = load_image("/home/user/new2D3D/characters/ref/front.png").convert("RGB").resize((512, 512))

cute_prompt = ("1girl, anime style, kawaii, moe, round soft face, huge sparkling round eyes, "
               "blonde twintails, blue eyes, plaid sweater, red scarf, green pleated skirt, "
               "cel shading, official character art, white background, best quality, masterpiece")
neg_prompt = ("lowres, bad anatomy, extra limbs, blurry, watermark, mature face, sharp jaw, "
              "long face, realistic, adult, narrow eyes")

variants = [
    dict(name="anythingv5_s030", model="hsuwill000/LCM-anything-v5-openvino", strength=0.30, clip_skip=0, steps=8),
    dict(name="anythingv5_s020", model="hsuwill000/LCM-anything-v5-openvino", strength=0.20, clip_skip=0, steps=8),
    dict(name="meinamix_s030",   model="iamanaiart/LCM-meinamix_meinaV11-openvino", strength=0.30, clip_skip=0, steps=8),
    dict(name="meinamix_s020",   model="iamanaiart/LCM-meinamix_meinaV11-openvino", strength=0.20, clip_skip=0, steps=8),
]

loaded = {}
for v in variants:
    if v["model"] not in loaded:
        print("loading", v["model"])
        t0 = time.time()
        loaded[v["model"]] = OVStableDiffusionImg2ImgPipeline.from_pretrained(v["model"], ov_config={"CACHE_DIR": "./ov_cache"})
        print(f"  load: {time.time()-t0:.1f}s")
    pipe = loaded[v["model"]]
    t0 = time.time()
    image = pipe(
        prompt=cute_prompt,
        negative_prompt=neg_prompt,
        image=init_image,
        strength=v["strength"],
        num_inference_steps=v["steps"],
        guidance_scale=1.5,
        generator=np.random.RandomState(42),
    ).images[0]
    dt = time.time() - t0
    print(f"{v['name']}: {dt:.2f}s")
    image.save(f"out_{v['name']}.png")
