"""IP-Adapter+ControlNet+MeinaMix 45度(正面と側面90度の中間角度)、v6。

目的は品質向上ではなく**連続性の確認**。v5(側面90度)は識別要素の
再現に成功したが、これは1角度・1シードの結果に過ぎず、正面(0度)から
側面(90度)まで滑らかに繋がる中間角度が同じ品質・同じキャラとして
出せるかは未検証だった。設定はv5から変更せず(ステップ数40・
IP-Adapter scale 0.8・識別要素プロンプト)、姿勢ガイドのみ45度用に
作り直して、角度以外の変数を揃えた比較を行う。

45度の姿勢ガイドは、正面検出キーポイント(両側とも可視)と側面プロファイル
キーポイント(近い側のみ可視、遠い側はNone)を50%で補間して作る。
遠い側は「Noneに向けて」ではなく「中心線に向けて」補間することで、
半分隠れた3/4視点らしいトポロジー(両肩は見えるが遠い側は中心寄り)にする。
"""
import os
import time

import cv2
import numpy as np
import torch
from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose.body import Keypoint
from controlnet_aux.open_pose.util import draw_bodypose
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
MASTER = os.path.join(ROOT, "characters", "zero", "master_v2_fullbody.png")
OUT_DIR = os.path.join(HERE, "spike_ipadapter")
os.makedirs(OUT_DIR, exist_ok=True)

# 側面(v2)で使ったのと同じペア: (front側indexA, front側indexB, 側面での近depth係数)
NEAR_PAIRS = [(1, 1.15), (15, 0.70), (17, -0.55), (3, 0.30), (4, 0.45), (5, 0.30),
              (9, 0.0), (10, 0.15), (11, 0.08)]
# 正面のみで検出され側面では隠れる(遠い)側のペア: front側index -> 対応する近indexの鏡
FAR_TO_NEAR = {6: 3, 7: 4, 8: 5, 12: 9, 13: 10, 14: 11, 16: 15, 18: 17}


def detect_front_keypoints():
    detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    master = Image.open(MASTER).convert("RGB")
    poses = detector.detect_poses(np.array(master))
    return poses[0].body.keypoints, master.size


def build_45_pose(front_kp, size):
    def y(i):
        return front_kp[i - 1].y

    shoulder_w = abs(front_kp[5].x - front_kp[2].x)
    xc = 0.5
    depth = shoulder_w * 0.55

    kp = [None] * 18

    # 近い側(v2の90度と同じdepth offset、y座標は正面の実測値のまま)。
    # index17(REar)は正面検出でNoneのため、index18(LEar)のyで代用する
    # (v2のbuild_side_profile_poseと同じ回避策)。
    for idx, dx in NEAR_PAIRS:
        src_idx = 18 if front_kp[idx - 1] is None else idx
        y_front = front_kp[src_idx - 1].y
        x_front = front_kp[src_idx - 1].x
        x_90 = xc + dx * depth
        x_45 = (x_front + x_90) / 2
        kp[idx - 1] = Keypoint(x=x_45, y=y_front)

    # 遠い側(正面のx→中心線へ50%寄せる。Noneではなく半分隠れた位置)
    for far_idx, near_idx in FAR_TO_NEAR.items():
        if front_kp[far_idx - 1] is None:
            continue
        x_front = front_kp[far_idx - 1].x
        y_front = front_kp[far_idx - 1].y
        x_45 = (x_front + xc) / 2
        kp[far_idx - 1] = Keypoint(x=x_45, y=y_front)

    # neck
    kp[1] = Keypoint(x=xc, y=front_kp[1].y)

    W, H = size
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas = draw_bodypose(canvas, kp)
    guide = Image.fromarray(canvas)
    guide.save(os.path.join(OUT_DIR, "pose_45deg_guide.png"))
    return guide


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
    front_kp, size = detect_front_keypoints()
    pose_45 = build_45_pose(front_kp, size)
    print("45deg pose guide built", flush=True)

    t0 = time.time()
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "Meina/MeinaMix_V11", controlnet=controlnet,
        torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(0.8)
    print(f"pipeline load: {time.time()-t0:.1f}s", flush=True)

    ip_image = Image.open(MASTER).convert("RGB")

    # v5と同一、"side view"のみ"three-quarter view"に変更(77トークン以内)
    prompt = ("best quality, anime girl, 1girl, solo, three-quarter view, "
              "silver white hair with pink and green streak highlights, "
              "ponytail, green eyes, black cat ear headset with mint "
              "green glow, white crop top, black shorts, white puffed "
              "sleeves, fingerless gloves, white sheer cape with teal "
              "trim, white thigh-high boots with teal accents, "
              "simple white background")
    neg = ("lowres, bad anatomy, blurry, watermark, text, realistic, 3d, "
           "extra fingers, missing fingers, deformed, cropped, cut off, "
           "multiple views, turnaround, colored background, scenery, tail, "
           "glowing ring, light ring, motion lines, brown eyes, red eyes, "
           "blue eyes, black hair, purple shoes")

    t0 = time.time()
    result = pipe(
        prompt=prompt, negative_prompt=neg, image=pose_45,
        ip_adapter_image=ip_image,
        num_inference_steps=40, guidance_scale=7.0,
        generator=torch.Generator().manual_seed(1),
    ).images[0]
    dt = time.time() - t0
    out_path = os.path.join(OUT_DIR, "calib_45deg_ipa_s1.png")
    result.save(out_path)
    print(f"generate: {dt:.1f}s -> {out_path}", flush=True)

    sim = identity_histogram_similarity(MASTER, result)
    print(f"identity histogram similarity (参考値、信頼性低いと判明済み): {sim:.3f}", flush=True)


if __name__ == "__main__":
    main()
