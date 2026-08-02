import time
from optimum.intel import OVStableDiffusionPipeline

model_id = "rupeshs/sd-turbo-openvino"
print("loading pipeline...")
t0 = time.time()
pipe = OVStableDiffusionPipeline.from_pretrained(model_id, ov_config={"CACHE_DIR": "./ov_cache"})
print(f"load time: {time.time()-t0:.1f}s")

prompt = "1girl, anime style, blonde twintails, blue eyes, simple background"
t0 = time.time()
image = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0, height=512, width=512).images[0]
dt = time.time() - t0
print(f"generation time (2 steps): {dt:.2f}s")
image.save("out_turbo_test.png")
