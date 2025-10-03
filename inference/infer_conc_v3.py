import os
import json
import yaml
import cv2
import numpy as np
import torch
from mmcv import Config
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmcv.ops import batched_nms

# ======== Configurable Parameters ========
PATCH_SIZE = 320
OVERLAP = 160
NMS_IOU = 0.3
USE_CLAHE = True

# ======== Preprocessing Functions ========
def apply_clahe(patch):
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def load_and_crop_image(image_path=None, image=None, patch_size=PATCH_SIZE, overlap=OVERLAP):
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
    elif image is not None:
        img = image
    else:
        raise ValueError("Must provide image_path or image.")

    # Optional padding to ensure border coverage
    pad = overlap
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    h, w, _ = img_padded.shape
    patches, coords = [], []
    for y in range(0, h - patch_size + 1, patch_size - overlap):
        for x in range(0, w - patch_size + 1, patch_size - overlap):
            patch = img_padded[y:y + patch_size, x:x + patch_size].copy()
            patches.append(patch)
            coords.append((x - pad, y - pad))  # compensate padding
    return patches, coords, img

def crop_center(image, crop_size=1000):           # tutaj na 320
    h, w, _ = image.shape
    y0 = max(0, (h - crop_size) // 2)
    x0 = max(0, (w - crop_size) // 2)
    return image[y0:y0 + crop_size, x0:x0 + crop_size].copy()

# ======== MMDet Custom Loader ========
from mmdet.datasets.builder import PIPELINES
@PIPELINES.register_module()
class LoadImageFromNDArray:
    def __call__(self, results):
        img = results['img']
        results.update({
            'filename': results.get('filename', None),
            'ori_filename': results.get('filename', None),
            'img': img,
            'img_shape': img.shape,
            'ori_shape': img.shape,
            'pad_shape': img.shape,
            'scale_factor': 1.0,
            'flip': False,
            'flip_direction': None,
            'img_fields': ['img']
        })
        return results

# ======== NMS ========
def approximate_global_nms(detections, iou_thr=0.1):
    if len(detections) == 0:
        return []
    bboxes, scores, labels = [], [], []
    for (box, score, label) in detections:
        cx, cy, w, h, angle = box
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        bboxes.append([x1, y1, x2, y2])
        scores.append(score)
        labels.append(label)
    bboxes = torch.tensor(np.array(bboxes, dtype=np.float32))
    scores = torch.tensor(np.array(scores, dtype=np.float32))
    labels = torch.tensor(np.array(labels, dtype=np.int64))
    nms_cfg = dict(type='nms', iou_threshold=iou_thr)
    _, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    return [detections[i] for i in keep]

# ======== Inference ========
def run_inference_on_patches(config_path, checkpoint_path, patches, coords, score_thr, device, save_patch_preds=False, patch_dir=None):
    cfg = Config.fromfile(config_path)
    model = init_detector(cfg, checkpoint_path, device=device)
    model.eval()
    pipeline = Compose([
        dict(type='LoadImageFromNDArray'),
        dict(type='Resize', img_scale=(PATCH_SIZE, PATCH_SIZE), keep_ratio=True),
        dict(type='Normalize', mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
    ])
    all_detections = []
    metadata = []
    print(f"Running inference on {len(patches)} patches...")
    for i, (patch, (x0, y0)) in enumerate(zip(patches, coords)):
        if USE_CLAHE:
            patch = apply_clahe(patch)
        data = dict(img=patch, filename=f'patch_{i}.jpg')
        data = pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device])[0]
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, img=[data['img']], img_metas=[data['img_metas']])[0]
        patch_detections = 0
        vis_patch = patch.copy()
        for label, class_bboxes in enumerate(result):
            for box in class_bboxes:
                score = box[-1]
                if score >= score_thr:
                    cx, cy, w, h, angle = box[:5]
                    cx += x0
                    cy += y0
                    all_detections.append(((cx, cy, w, h, angle), score, label))
                # Draw all boxes for visualization
                cx_v, cy_v, w_v, h_v, angle_v = box[:5]
                rect = ((cx_v, cy_v), (w_v, h_v), np.degrees(angle_v))
                pts = cv2.boxPoints(rect).astype(np.int32)
                cv2.polylines(vis_patch, [pts], True, (0,0,255), 1)
                patch_detections += 1
        metadata.append({"patch": i, "x": int(x0), "y": int(y0), "detections": patch_detections})
        if save_patch_preds:
            cv2.imwrite(os.path.join(patch_dir, f"patch_{i}_preds.jpg"), vis_patch)
    print(f"Detected {len(all_detections)} boxes before NMS.")
    all_detections = approximate_global_nms(all_detections)
    print(f"{len(all_detections)} boxes after NMS.")
    return all_detections, metadata

def draw_detections(image, detections):
    for (box, score, label) in detections:
        cx, cy, w, h, angle = box
        rect = ((cx, cy), (w, h), np.degrees(angle))
        pts = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(image, [pts], True, (0,255,0), 1)
    return image

# ======== Main ========
def main():
    image_path = input("Enter image filename: ").strip()
    if not os.path.isfile(image_path):
        print("File not found.")
        return

    config_path = "../configs/custom/my_rotated_retinanet_obb_p2_precision.py"
    checkpoint_path = "best_mAP_epoch_32.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_full = cv2.imread(image_path)
    h_full, w_full, _ = img_full.shape
    print(f"\nImage size: {w_full} x {h_full}")

    mode = input("Select mode:\n1) Full image\n2) Central 1000x1000 crop\n> ").strip()
    if mode == "2":
        img_proc = crop_center(img_full)
        volume_pixels = 1_000_000
        img_w, img_h = 1000, 1000
    else:
        img_proc = img_full
        volume_pixels = (4200-400)*(3120-400)
        img_w, img_h = w_full, h_full

    score_thr = float(input("Detection threshold (e.g., 0.3): ").strip())
    dilution_factor = float(input("Dilution factor (e.g., 11): ").strip())
    export_ls = input("Export to Label Studio? (y/n): ").strip().lower() == "y"
    visualize_patches = input("Save per-patch images? (y/n): ").strip().lower() == "y"

    img_base = os.path.splitext(os.path.basename(image_path))[0]
    result_dir = f"results_{img_base}"
    os.makedirs(result_dir, exist_ok=True)
    if visualize_patches:
        os.makedirs(os.path.join(result_dir, "patch_preds"), exist_ok=True)

    if mode == "2":
        cv2.imwrite(os.path.join(result_dir, "cropped.png"), img_proc)

    patches, coords, _ = load_and_crop_image(image=img_proc)
    detections, metadata = run_inference_on_patches(
        config_path, checkpoint_path, patches, coords, score_thr, device,
        save_patch_preds=visualize_patches,
        patch_dir=os.path.join(result_dir, "patch_preds")
    )
    result_image = draw_detections(img_proc.copy(), detections)
    cv2.imwrite(os.path.join(result_dir, "annotated.jpg"), result_image)

    with open(os.path.join(result_dir, "detections.txt"), "w") as f:
        for (box, score, label) in detections:
            cx, cy, w, h, angle = box
            f.write(f"{cx:.2f},{cy:.2f},{w:.2f},{h:.2f},{angle:.4f},{score:.4f},{label}\n")

    with open(os.path.join(result_dir, "patch_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(result_dir, "detections.yaml"), "w") as f:
        yaml.dump([{
            "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h),
            "angle": float(angle), "score": float(score), "label": int(label)
        } for (box, score, label) in detections], f)

    pixel_size_um = 1.1
    depth_um = 20
    volume_um3 = volume_pixels * (pixel_size_um**2) * depth_um
    volume_ml = volume_um3 * 1e-12
    N = len(detections)
    concentration = (N / volume_ml) * (dilution_factor / 1e6)

    print("\n=== Results ===")
    print(f"Detected: {N}")
    print(f"Dilution: {dilution_factor}")
    print(f"Volume (ml): {volume_ml:.3e}")
    print(f"Concentration: {concentration:.2f} million/ml")

    if export_ls:
        ls_annotations = []
        for (box, score, label) in detections:
            cx, cy, w, h, angle = box
            rect = ((cx, cy), (w, h), np.degrees(angle))
            pts = cv2.boxPoints(rect)
            points = [[(x / img_w) * 100, (y / img_h) * 100] for x, y in pts]
            ls_annotations.append({
                "original_width": img_w,
                "original_height": img_h,
                "image_rotation": 0,
                "value": {"points": points, "polygonlabels": ["sperm"]},
                "from_name": "label",
                "to_name": "image",
                "type": "polygonlabels"
            })
        labelstudio_data = [{
            "data": {"image": os.path.join(result_dir, "annotated.jpg")},
            "annotations": [{"result": ls_annotations}]
        }]
        with open(os.path.join(result_dir, "labelstudio.json"), "w") as f:
            json.dump(labelstudio_data, f, indent=2)

    with open(os.path.join(result_dir, "README.txt"), "w") as f:
        f.write("This folder contains:\n")
        f.write("- annotated.jpg: detections drawn\n")
        f.write("- detections.txt: raw detections\n")
        f.write("- detections.yaml: same in YAML\n")
        f.write("- patch_metadata.json: per-patch info\n")
        f.write("- labelstudio.json: Label Studio annotations\n")

if __name__ == "__main__":
    main()
