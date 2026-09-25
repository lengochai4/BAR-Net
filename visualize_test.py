import os
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom

import argparse
from config import get_config
from networks.vision_transformer import SwinUnet

def main():
    # Setup args & config
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml')
    parser.add_argument('--opts', default=None, nargs='+')
    parser.add_argument('--dataset', type=str, default='Synapse')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--cache-mode', type=str, default='part')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--amp-opt-level', type=str, default='O1')
    parser.add_argument('--tag', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--throughput', action='store_true')
    args, _ = parser.parse_known_args()
    config = get_config(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    model = SwinUnet(config, img_size=224, num_classes=9).to(device)
    model.eval()

    weight_path = "checkpoints/best_model.pth"
    if not os.path.exists(weight_path):
        print(f"Error: {weight_path} not found!")
        return

    print("Loading weights...")
    state_dict = torch.load(weight_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    # Load test file
    test_file = "data_test/case0001.npy.h5"
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found!")
        return

    print(f"Reading {test_file}...")
    with h5py.File(test_file, 'r') as f:
        image = f['image'][:]
        label = f['label'][:] if 'label' in f else None

    # We will visualize the middle slice
    mid_slice = image.shape[0] // 2
    if 'frame02' in test_file:
         mid_slice = image.shape[0] // 2
    slice_2d = image[mid_slice]
    gt_2d = label[mid_slice] if label is not None else None

    x, y = slice_2d.shape
    if x != 224 or y != 224:
        model_input = zoom(slice_2d, (224 / x, 224 / y), order=3)
    else:
        model_input = slice_2d

    img_tensor = torch.from_numpy(model_input).unsqueeze(0).unsqueeze(0).float().to(device)

    print("Running inference...")
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
        if x != 224 or y != 224:
            pred = zoom(pred, (x / 224, y / 224), order=0)

    # Visualization
    print("Saving visualization to output_vis.png...")
    COLORS = np.array([
        [0, 0, 0, 0],           # Background
        [230, 80, 80, 200],     # Class 1
        [100, 150, 210, 200],   # Class 2
        [80, 180, 100, 200],    # Class 3
        [240, 230, 90, 200],    # Class 4
        [255, 0, 255, 200],     # Class 5
        [0, 255, 255, 200],     # Class 6
        [128, 0, 0, 200],       # Class 7
        [0, 128, 0, 200],       # Class 8
    ], dtype=np.uint8)

    # Base image for plotting
    s_min, s_max = slice_2d.min(), slice_2d.max()
    vis_base = (slice_2d - s_min) / (s_max - s_min + 1e-8)
    vis_base = (vis_base * 255).astype(np.uint8)
    base_img = Image.fromarray(vis_base).convert("RGBA")

    # Pred mask
    pred_rgba = COLORS[pred]
    pred_img = Image.fromarray(pred_rgba, mode='RGBA')

    # Ground truth mask (if available)
    if gt_2d is not None:
        gt_rgba = COLORS[gt_2d.astype(int)]
        gt_img = Image.fromarray(gt_rgba, mode='RGBA')
    else:
        gt_img = None

    fig, axes = plt.subplots(1, 3 if gt_img else 2, figsize=(15 if gt_img else 10, 5))
    
    # Original
    axes[0].imshow(base_img)
    axes[0].set_title("CT Input")
    axes[0].axis('off')

    # Prediction
    pred_blend = Image.alpha_composite(base_img, pred_img)
    if gt_img:
        axes[1].imshow(Image.alpha_composite(base_img, gt_img))
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(pred_blend)
        axes[2].set_title("BAR-Net Prediction")
        axes[2].axis('off')
    else:
        axes[1].imshow(pred_blend)
        axes[1].set_title("BAR-Net Prediction")
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("output_vis.png", dpi=150)
    print("Done! Check output_vis.png")
    
    # Print statistics
    unique, counts = np.unique(pred, return_counts=True)
    print("Predicted Classes & Pixel Counts:")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} pixels")

if __name__ == "__main__":
    main()
