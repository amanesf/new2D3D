import time
import numpy as np
from PIL import Image
from optimum.intel import OVStableDiffusionImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler

face_crop = Image.open("face_crop_raw.png").convert("RGB").resize((768, 768))

prompt = ("masterpiece, best quality, 1girl, anime style, extremely kawaii, super cute face, "
          "close-up portrait, huge sparkling round eyes, long eyelashes, rosy cheeks, "
          "soft round baby face, blonde twintails, green hair ribbon, blue scarf-red scarf, "
          "cheerful big smile, official character art, clean lineart, cel shading, "
          "white background, vibrant colors")
neg_prompt = ("lowres, bad anatomy, extra limbs, blurry, watermark, mature face, sharp jaw, "
              "long face, realistic, adult, narrow eyes, closed mouth, serious expression, "
              "asymmetric eyes, deformed, ugly, cropped face, out of frame")

print("loading MeinaMix V11 (full quality, OpenVINO)...")
t0 = time.time()
pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained("./meinamix_v11_ov", ov_config={"CACHE_DIR": "./ov_cache_v11"})
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
print(f"load: {time.time()-t0:.1f}s")

for seed in [1, 2, 3, 4]:
    t0 = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=face_crop,
        strength=0.55,
        num_inference_steps=40,
        guidance_scale=7.5,
        generator=np.random.RandomState(seed),
    ).images[0]
    dt = time.time() - t0
    print(f"seed={seed}: {dt:.2f}s")
    image.save(f"out_face_cute_seed{seed}.png")
