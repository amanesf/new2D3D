import time
import numpy as np
from optimum.intel import OVStableDiffusionImg2ImgPipeline
from diffusers.utils import load_image

init_image = load_image("/home/user/new2D3D/characters/ref/front.png").convert("RGB").resize((512, 512))

cute_prompt = ("1girl, anime style, kawaii, moe, round soft face, huge sparkling round eyes, "
               "blonde twintails, green hair ribbon, blue eyes, plaid sweater, red scarf, green pleated skirt, "
               "cheerful smile, cel shading, official character art, white background, best quality, masterpiece")
neg_prompt = ("lowres, bad anatomy, extra limbs, blurry, watermark, mature face, sharp jaw, "
              "long face, realistic, adult, narrow eyes, closed mouth, serious expression")

model = "iamanaiart/LCM-meinamix_meinaV11-openvino"
print("loading", model)
pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(model, ov_config={"CACHE_DIR": "./ov_cache"})

variants = [
    dict(name="meinamix_s022_steps10", strength=0.22, steps=10),
    dict(name="meinamix_s025_steps12", strength=0.25, steps=12),
]
for v in variants:
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
