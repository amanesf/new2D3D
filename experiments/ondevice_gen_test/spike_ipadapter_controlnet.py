"""IP-Adapter+ControlNet+MeinaMixでの頭部/全身側面ビュー実現性スパイク
(校正1枚、側面90度)。

前回のControlNet単体スパイク(spike_controlnet_head_rotation.py)は
同一性ゲート0.456でFAILし、識別要素を画像で固定する仕組みが無いこと・
姿勢ガイドが「簡易横圧縮」で実際の側面骨格として不正確だったことが
主因と特定した。本スクリプトはその2点を修正する:

1. IP-Adapter(h94/IP-Adapter, SD1.5)でマスター画像から識別要素を
   画像として固定する
2. 姿勢ガイドは正面検出キーポイントの縦方向(身長方向)の実測値を保持し、
   横方向のみ「体の奥行き」として作り直した、片側のみ可視な
   本物の側面骨格トポロジーを使う(前回のような両肩・両腰を残したまま
   横に潰す方式ではない)
"""
import os
import time

import cv2
import numpy as np
import torch
from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose.body import BodyResult, Keypoint
from controlnet_aux.open_pose.util import draw_bodypose
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
MASTER = os.path.join(ROOT, "characters", "zero", "master_v2_fullbody.png")
OUT_DIR = os.path.join(HERE, "spike_ipadapter")
os.makedirs(OUT_DIR, exist_ok=True)


def detect_front_keypoints():
    detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    master = Image.open(MASTER).convert("RGB")
    poses = detector.detect_poses(np.array(master))
    assert poses, "front pose detection failed"
    return poses[0].body.keypoints, master.size  # (W, H)


def build_side_profile_pose(front_kp, size):
    """正面検出の縦方向(身長方向)実測値を保持しつつ、横方向は片側のみ
    可視な側面プロファイルとして作り直す。両肩・両腰を横に潰すのではなく、
    片方の肩・肘・手首・腰・膝・足首の鎖のみを定義し、反対側はNoneにする
    ことで、実際の側面骨格トポロジーにする。
    """
    def y(i):
        return front_kp[i - 1].y

    y_nose, y_eye, y_ear = y(1), y(15), y(18)
    y_neck = y(2)
    y_shoulder = (y(3) + y(6)) / 2
    y_elbow = (y(4) + y(7)) / 2
    y_wrist = (y(5) + y(8)) / 2
    y_hip = (y(9) + y(12)) / 2
    y_knee = (y(10) + y(13)) / 2
    y_ankle = (y(11) + y(14)) / 2

    shoulder_w = abs(front_kp[5].x - front_kp[2].x)  # LShoulder.x - RShoulder.x
    xc = 0.5  # 画面中央に立たせる(前回と同条件)
    depth = shoulder_w * 0.55  # 奥行きは肩幅の55%程度(アニメ体型の目安)

    kp = [None] * 18

    def K(idx, dx, y_):
        kp[idx - 1] = Keypoint(x=xc + dx * depth, y=y_)

    # 顔(片側のみ、鼻は進行方向に突き出す)
    K(1, 1.15, y_nose)   # Nose
    K(15, 0.70, y_eye)   # REye(可視側の目として流用)
    K(17, -0.55, y_ear)  # REar(後頭部寄り)

    # 体幹
    K(2, 0.0, y_neck)    # Neck
    K(3, 0.30, y_shoulder)   # RShoulder(可視側)
    K(4, 0.45, y_elbow)      # RElbow
    K(5, 0.30, y_wrist)      # RWrist(体側に垂らす)
    K(9, 0.0, y_hip)         # RHip
    K(10, 0.15, y_knee)      # RKnee
    K(11, 0.08, y_ankle)     # RAnkle
    # LShoulder/LElbow/LWrist/LHip/LKnee/LAnkle/LEye/LEarは反対側で
    # 隠れるためNoneのまま(draw_bodyposeはNoneの限を自動でスキップする)

    W, H = size
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas = draw_bodypose(canvas, kp)
    guide = Image.fromarray(canvas)
    guide.save(os.path.join(OUT_DIR, "pose_side90_guide_v2.png"))
    return guide


def identity_histogram_similarity(ref_path, gen_img):
    """同一性の簡易指標(色ヒストグラム類似度、前回スパイクと同じ手法)。
    厳密な同一性判定ではなく、破綻の粗いスクリーニング用。
    """
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
    pose_side = build_side_profile_pose(front_kp, size)
    print("pose guide built", flush=True)

    t0 = time.time()
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "Meina/MeinaMix_V11", controlnet=controlnet,
        torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(0.65)
    print(f"pipeline load: {time.time()-t0:.1f}s", flush=True)

    ip_image = Image.open(MASTER).convert("RGB")

    prompt = ("masterpiece, best quality, anime girl, 1girl, solo, side view, "
              "kawaii, clean lineart, cel shading, simple white background")
    neg = ("lowres, bad anatomy, blurry, watermark, text, realistic, 3d, "
           "extra fingers, missing fingers, deformed, cropped, cut off, "
           "multiple views, turnaround, colored background, scenery")

    t0 = time.time()
    result = pipe(
        prompt=prompt, negative_prompt=neg, image=pose_side,
        ip_adapter_image=ip_image,
        num_inference_steps=20, guidance_scale=7.0,
        generator=torch.Generator().manual_seed(1),
    ).images[0]
    dt = time.time() - t0
    out_path = os.path.join(OUT_DIR, "calib_side90_ipa_s1.png")
    result.save(out_path)
    print(f"generate: {dt:.1f}s -> {out_path}", flush=True)

    sim = identity_histogram_similarity(MASTER, result)
    print(f"identity histogram similarity: {sim:.3f}", flush=True)


if __name__ == "__main__":
    main()
