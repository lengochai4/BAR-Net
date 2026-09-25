from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import numpy as np
import io
from PIL import Image
import base64
import sys
import os

# Adds the parent directory to the system path so we can import the model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from networks.vision_transformer import SwinUnet
from config import get_config
import argparse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Configuration
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

yaml_path = os.path.join(parent_dir, "configs", "swin_tiny_patch4_window7_224_lite.yaml")
if os.path.exists(yaml_path):
    args.cfg = yaml_path
config = get_config(args)

# Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SwinUnet(config, img_size=224, num_classes=9).to(device)
model.eval()

weight_path = os.path.join(parent_dir, "checkpoints", "best_model.pth")
if os.path.exists(weight_path):
    print(f"Loading weights from {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
        
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if wrapped in DataParallel
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    try:
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print("Weights loaded! Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
        if len(missing) > 0:
            print("First 5 missing:", missing[:5])
    except Exception as e:
        print(f"Warning: Could not completely load weights. Error: {e}")
else:
    print(f"Weights not found, using random init!")

# Colors for 9 classes to display the mask (RGBA Format)
COLORS = np.array([
    [0, 0, 0, 0],           # Background - Fully Transparent
    [230, 80, 80, 200],     # Class 1 - Aorta (Red/Coral)
    [100, 150, 210, 200],   # Class 2 - Left Kidney (Blue)
    [80, 180, 100, 200],    # Class 3 - Right Kidney (Green)
    [240, 230, 90, 200],    # Class 4 - Liver (Yellow/Gold)
    [255, 0, 255, 200],     # Class 5 - Magenta
    [0, 255, 255, 200],     # Class 6 - Cyan
    [128, 0, 0, 200],       # Class 7 - Maroon
    [0, 128, 0, 200],       # Class 8 - Olive
], dtype=np.uint8)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Convert to Grayscale (1-channel) to match training data
        image = Image.open(io.BytesIO(contents)).convert("L")
        orig_size = image.size
        
        # Resize to 224x224
        image_resized = image.resize((224, 224), Image.BILINEAR)
        
        # Convert to numpy and normalize identically to h5 (usually 0-1 or min-max if needed, but since it's an image, 0-1 is safe natively)
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        
        from scipy.ndimage import zoom
        # No ImageNet normalization! Just raw 1-channel float values, replicated internally by SwinUnet
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            
            # Usually output is logits [batch, classes, height, width]
            probs = F.softmax(output, dim=1)
            prediction = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy() # [224, 224]

        # Create RGBA base from original (full-size) image
        orig_img_rgba = Image.fromarray(np.array(image_resized, dtype=np.uint8)).convert("RGBA").resize(orig_size, Image.BILINEAR)
        
        # Convert prediction to RGBA Image (at 224x224 prediction)
        mask_rgba = COLORS[prediction.astype(int)]
        mask_only_img = Image.fromarray(mask_rgba, mode='RGBA').resize(orig_size, Image.NEAREST)
        
        # BAKE THE MASK ON TOP OF ORIGINAL IMAGE IN PYTHON (same size now)
        blended_image = Image.alpha_composite(orig_img_rgba, mask_only_img)
        
        # Encode blended image to base64
        buffered = io.BytesIO()
        blended_image.convert("RGB").save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {"status": "success", "mask": mask_base64}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

import h5py
import tempfile
import uuid

@app.post("/predict_volume")
async def predict_volume(file: UploadFile = File(...)):
    try:
        # HDF5 needs a file on disk, so we save it temporarily
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_vol.h5")
        
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        original_slices = []
        mask_slices = []
        
        with h5py.File(temp_path, 'r') as data:
            if 'image' not in data:
                raise ValueError("File does not contain 'image' dataset.")
                
            vol = data['image'][:]  # shape: [slices, H, W]
            
            from scipy.ndimage import zoom
            
            total_slices = vol.shape[0]
            sampled_indices = list(range(total_slices))
            print(f"[INFO] Volume has {total_slices} slices, processing all")
            
            # Predict slice by slice
            for idx, i in enumerate(sampled_indices):
                slice_2d = vol[i, :, :]
                
                # --- 1. Prepare raw slice FOR MODEL (just like test.py) ---
                x, y = slice_2d.shape[0], slice_2d.shape[1]
                if x != 224 or y != 224:
                    model_input_slice = zoom(slice_2d, (224 / x, 224 / y), order=3)
                else:
                    model_input_slice = slice_2d
                    
                img_tensor = torch.from_numpy(model_input_slice).unsqueeze(0).unsqueeze(0).float().to(device)
                
                # Inference
                with torch.no_grad():
                    output = model(img_tensor)
                    probs = F.softmax(output, dim=1)
                    prediction = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
                
                # Zoom prediction back to original resolution OUTSIDE torch context
                if x != 224 or y != 224:
                    prediction = zoom(prediction, (x / 224, y / 224), order=0)
                prediction = prediction.astype(int)
                
                # DEBUG: Print predicted classes for first and middle slice
                mid_idx = len(sampled_indices) // 2
                if idx == 0 or idx == mid_idx:
                    unique_classes = np.unique(prediction)
                    print(f"[DEBUG] Slice index {i} (sample {idx}): unique classes = {unique_classes}")
                
                # --- 2. Prepare visual slices FOR FRONTEND ---
                s_min, s_max = slice_2d.min(), slice_2d.max()
                vis_slice = (slice_2d - s_min) / (s_max - s_min + 1e-8)
                vis_slice = (vis_slice * 255).astype(np.uint8)  # shape: (x, y)
                orig_pil = Image.fromarray(vis_slice, mode='L').convert("RGB")
                
                # Clamp prediction values to valid range [0, 8]
                pred_clamped = np.clip(prediction, 0, len(COLORS) - 1)
                mask_rgba = COLORS[pred_clamped]           # shape: (x, y, 4)
                mask_pil = Image.fromarray(mask_rgba, mode='RGBA')
                
                # Convert original to base64
                buf_orig = io.BytesIO()
                orig_pil.save(buf_orig, format="PNG")
                original_slices.append(base64.b64encode(buf_orig.getvalue()).decode('utf-8'))
                
                # Convert MASK (RGBA with transparent background) to base64
                buf_mask = io.BytesIO()
                mask_pil.save(buf_mask, format="PNG")
                mask_slices.append(base64.b64encode(buf_mask.getvalue()).decode('utf-8'))
                
        # Clean up temp file
        os.remove(temp_path)
        
        return {
            "status": "success", 
            "original_slices": original_slices,
            "mask_slices": mask_slices,
            "num_slices": len(original_slices)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# Serve static files with absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
