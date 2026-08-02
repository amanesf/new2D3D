import time
from optimum.intel import OVStableDiffusionImg2ImgPipeline
from diffusers.utils import load_image

model_id = "hsuwill000/LCM-anything-v5-openvino"
print("loading pipeline...")
t0 = time.time()
pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(model_id, ov_config={"CACHE_DIR": "./ov_cache"})
print(f"load time: {time.time()-t0:.1f}s")

init_image = load_image("/home/user/new2D3D/characters/ref/front.png").convert("RGB").resize((512, 512))

prompt = "1girl, anime style, blonde twintails, blue eyes, plaid sweater, red scarf, green pleated skirt, white background, best quality"
neg_prompt = "lowres, bad anatomy, extra limbs, blurry, watermark"

for strength in [0.3, 0.5]:
    t0 = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=8,
        guidance_scale=1.5,
    ).images[0]
    dt = time.time() - t0
    print(f"strength={strength} time: {dt:.2f}s")
    image.save(f"out_img2img_s{strength}.png")
