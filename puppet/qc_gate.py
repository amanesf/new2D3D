#!/usr/bin/env python3
"""QCゲート(SUPER_LIVE2D_V3_PLAN.md §2検証プロトコルのコード化)。

「Claudeの目」を全景の目視や集計値だけに頼らせず、以下を機械的に強制する:
1. 境界コンタクトシート: 全パラメータ極値×境界のズームクロップをタイル化
2. 真のゴースト検出: 静止時の元画素が、動いたレイヤーのアルファで
   覆われずに露出していないか
3. 隙間検出: 接続ゾーン(根元付近、静止時は連続した絵柄であるべき領域)で、
   動かした結果、背景色(隙間)が新たに露出していないか

このスクリプトは「合格/不合格」をプロセスの出口として明示する。
不合格ならこのスクリプトが失敗を返す — 目視の印象で「継ぎ目なし」と
申告することを禁止する運用(検証プロトコル§2-5)の実体。
"""
import json
import os
import sys

import cv2
import numpy as np


def is_nonwhite(img, thresh=245):
    return (img[:, :, 0] < thresh) | (img[:, :, 1] < thresh) | (img[:, :, 2] < thresh)


def rotate_sprite(sprite_rgba, pivot_local, angle_deg, pad=80):
    h, w = sprite_rgba.shape[:2]
    src_pad = cv2.copyMakeBorder(sprite_rgba, pad, pad, pad, pad,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    M = cv2.getRotationMatrix2D((pivot_local[0] + pad, pivot_local[1] + pad), angle_deg, 1.0)
    rotated = cv2.warpAffine(src_pad, M, (w + 2 * pad, h + 2 * pad),
                              flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    return rotated, pad


def place_on_canvas(canvas_shape, rotated, pad, place_xy):
    H, W = canvas_shape[:2]
    full_alpha = np.zeros((H, W), np.uint8)
    full_bgr = np.zeros((H, W, 3), np.uint8)
    rx0, ry0 = place_xy[0] - pad, place_xy[1] - pad
    x0c, y0c = max(0, rx0), max(0, ry0)
    x1c, y1c = min(W, rx0 + rotated.shape[1]), min(H, ry0 + rotated.shape[0])
    if x1c <= x0c or y1c <= y0c:
        return full_bgr, full_alpha
    sx0, sy0 = x0c - rx0, y0c - ry0
    seg = rotated[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
    full_bgr[y0c:y1c, x0c:x1c] = seg[:, :, :3]
    full_alpha[y0c:y1c, x0c:x1c] = seg[:, :, 3]
    return full_bgr, full_alpha


def composite(base_bgr, full_bgr, full_alpha):
    a = (full_alpha[:, :, None].astype(np.float32)) / 255.0
    out = base_bgr.astype(np.float32) * (1 - a) + full_bgr.astype(np.float32) * a
    return out.astype(np.uint8)


def check_ghost(base_holed_bgr, static_footprint_mask, moved_alpha, alpha_thresh=25):
    """真のゴースト: 元の静的footprintの中で、baseが非白(=まだ穴埋めできて
    いない/元絵が残っている)かつ動いたレイヤーのアルファで覆われていない画素。"""
    base_nonwhite = is_nonwhite(base_holed_bgr)
    uncovered = moved_alpha <= alpha_thresh
    ghost = (static_footprint_mask > 0) & base_nonwhite & uncovered
    return ghost


def check_gap(connection_zone_mask, original_silhouette_mask, result_bgr, moved_alpha,
              static_layers_alpha=None, alpha_thresh=25):
    """隙間: 接続ゾーン内で、静止時は絵(非白)だった場所が、結果画像でも
    どのレイヤーにも覆われず露出した背景(白)になっていないか。"""
    was_art = (connection_zone_mask > 0) & (original_silhouette_mask > 0)
    covered = moved_alpha > alpha_thresh
    if static_layers_alpha is not None:
        covered = covered | (static_layers_alpha > alpha_thresh)
    result_nonwhite = is_nonwhite(result_bgr)
    gap = was_art & (~covered) & (~result_nonwhite)
    return gap


def zoom_crop(img, box, scale):
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                       interpolation=cv2.INTER_NEAREST)


def tile_images(images, cols):
    rows = (len(images) + cols - 1) // cols
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    sheet = np.full((h * rows + 4 * (rows - 1), w * cols + 4 * (cols - 1), 3), 255, np.uint8)
    for i, im in enumerate(images):
        r, c = i // cols, i % cols
        y0, x0 = r * (h + 4), c * (w + 4)
        sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return sheet


def run_qc(config_path, out_dir):
    """config: JSON describing base/holed, sprite, pivot, angles to test,
    static footprint mask, connection zone mask, boundary crop boxes."""
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    base_dir = os.path.dirname(config_path)

    def load(p, flags=cv2.IMREAD_COLOR):
        return cv2.imread(os.path.join(base_dir, p), flags)

    base_holed = load(cfg["base_holed"])
    sprite = load(cfg["sprite"], cv2.IMREAD_UNCHANGED)
    place_xy = cfg["sprite_place_xy"]
    pivot_local = cfg["pivot_local"]
    static_footprint = load(cfg["static_footprint_mask"], cv2.IMREAD_GRAYSCALE)
    connection_zone = load(cfg["connection_zone_mask"], cv2.IMREAD_GRAYSCALE) \
        if cfg.get("connection_zone_mask") else None
    original_master = load(cfg["original_master"])
    original_silhouette = is_nonwhite(original_master).astype(np.uint8) * 255
    angles = cfg["test_angles"]
    boundary_boxes = cfg["boundary_crop_boxes"]  # list of [x0,y0,x1,y1]

    os.makedirs(out_dir, exist_ok=True)
    all_pass = True
    report_lines = []
    tiles = []

    for angle in angles:
        rotated, pad = rotate_sprite(sprite, pivot_local, angle)
        full_bgr, full_alpha = place_on_canvas(base_holed.shape, rotated, pad, place_xy)
        result = composite(base_holed, full_bgr, full_alpha)

        ghost = check_ghost(base_holed, static_footprint, full_alpha)
        ghost_n = int(ghost.sum())

        gap_n = 0
        if connection_zone is not None:
            gap = check_gap(connection_zone, original_silhouette, result, full_alpha)
            gap_n = int(gap.sum())

        ok = ghost_n <= cfg.get("ghost_tolerance_px", 50) and gap_n <= cfg.get("gap_tolerance_px", 50)
        all_pass = all_pass and ok
        report_lines.append(f"angle={angle:+d}: ghost_px={ghost_n} gap_px={gap_n} "
                             f"{'PASS' if ok else 'FAIL'}")

        out_path = os.path.join(out_dir, f"result_{angle:+d}.png")
        cv2.imwrite(out_path, result)

        for bi, box in enumerate(boundary_boxes):
            crop = zoom_crop(result, box, cfg.get("zoom_scale", 3))
            label = f"a={angle:+d} b{bi}"
            cv2.putText(crop, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            tiles.append(crop)

    sheet = tile_images(tiles, cols=len(boundary_boxes))
    sheet_path = os.path.join(out_dir, "_contact_sheet.png")
    cv2.imwrite(sheet_path, sheet)

    report = "\n".join(report_lines)
    report += f"\n\nOVERALL: {'PASS' if all_pass else 'FAIL'}"
    report += f"\ncontact sheet: {sheet_path}"
    with open(os.path.join(out_dir, "_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    return all_pass


if __name__ == "__main__":
    config_path = sys.argv[1]
    out_dir = sys.argv[2]
    ok = run_qc(config_path, out_dir)
    sys.exit(0 if ok else 1)
