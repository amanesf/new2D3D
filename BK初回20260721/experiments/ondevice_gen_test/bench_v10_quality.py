import time
import numpy as np
from optimum.intel import OVStableDiffusionImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import load_image

init_image = load_image("/home/user/new2D3D/characters/ref/front.png").convert("RGB").resize((512, 512))

cute_prompt = ("1girl, anime style, kawaii, moe, round soft face, huge sparkling round eyes, "
               "blonde twintails, green hair ribbon, blue eyes, plaid sweater, red scarf, green pleated skirt, "
               "cheerful smile, cel shading, official character art, white background, best quality, masterpiece")
neg_prompt = ("lowres, bad anatomy, extra limbs, blurry, watermark, mature face, sharp jaw, "
              "long face, realistic, adult, narrow eyes, closed mouth, serious expression")

print("loading full-step MeinaMix V10 (non-LCM, OpenVINO)...")
pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained("./meinamix_v10_ov", ov_config={"CACHE_DIR": "./ov_cache_v10"})
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

# test clip_skip support first
try:
    _ = pipe(prompt="test", num_inference_steps=1, image=init_image, strength=0.1, clip_skip=2, guidance_scale=1.0).images[0]
    print("clip_skip=2 SUPPORTED on this export")
    clip_skip_ok = True
except Exception as e:
    print("clip_skip=2 NOT supported:", repr(e))
    clip_skip_ok = False

variants = [
    dict(name="v10_full_s030_steps25_cfg7", strength=0.30, steps=25, cfg=7.0),
    dict(name="v10_full_s022_steps25_cfg7", strength=0.22, steps=25, cfg=7.0),
]
for v in variants:
    kwargs = dict(
        prompt=cute_prompt,
        negative_prompt=neg_prompt,
        image=init_image,
        strength=v["strength"],
        num_inference_steps=v["steps"],
        guidance_scale=v["cfg"],
        generator=np.random.RandomState(42),
    )
    if clip_skip_ok:
        kwargs["clip_skip"] = 2
    t0 = time.time()
    image = pipe(**kwargs).images[0]
    dt = time.time() - t0
    print(f"{v['name']}: {dt:.2f}s")
    image.save(f"out_{v['name']}.png")
