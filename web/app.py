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

from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 9 # Modify this based on your model/weights

model = SwinTransformerSys(
    img_size=224,
    num_classes=num_classes,
    embed_dim=96,
    depths=[2, 2, 2, 2],
    depths_decoder=[1, 2, 2, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
).to(device)
model.eval()

# Load weights if available inside checkpoints/ directory
weight_path = os.path.join(parent_dir, "checkpoints", "best_model.pth")
if os.path.exists(weight_path):
    print(f"Loading weights from {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    
    # Automatically remove 'module.' and 'swin_unet.' prefixes 
    # to perfectly match SwinTransformerSys standalone module.
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        name = name[10:] if name.startswith('swin_unet.') else name
        new_state_dict[name] = v
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Weights loaded perfectly seamlessly!")
    except Exception as e:
        print(f"Warning: Could not completely load weights. Error: {e}")
else:
    print(f"Weights not found at {weight_path}, using random init!")

# Colors for 9 classes to display the mask
COLORS = np.array([
    [0, 0, 0],         # Background - Black
    [255, 0, 0],       # Class 1 - Red
    [0, 255, 0],       # Class 2 - Green
    [0, 0, 255],       # Class 3 - Blue
    [255, 255, 0],     # Class 4 - Yellow
    [255, 0, 255],     # Class 5 - Magenta
    [0, 255, 255],     # Class 6 - Cyan
    [128, 0, 0],       # Class 7 - Maroon
    [0, 128, 0],       # Class 8 - Olive
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess Image
        orig_size = image.size
        # Resize to 224x224
        image_resized = image.resize((224, 224), Image.BILINEAR)
        img_array = np.array(image_resized) / 255.0
        
        # Normalize (Assuming standard ImageNet normalization, adjust if different for your model)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            
            # Usually output is logits [batch, classes, height, width]
            probs = F.softmax(output, dim=1)
            prediction = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy() # [224, 224]

        # Convert prediction to RGB Image
        mask_rgb = COLORS[prediction].astype(np.uint8)
        mask_image = Image.fromarray(mask_rgb).resize(orig_size, Image.NEAREST)
        
        # Encode mask to base64 to send back to frontend
        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")
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
            
            # Predict slice by slice
            for i in range(vol.shape[0]):
                slice_2d = vol[i, :, :]
                
                # Normalize origin slice for viewing
                s_min, s_max = slice_2d.min(), slice_2d.max()
                vis_slice = (slice_2d - s_min) / (s_max - s_min + 1e-8)
                vis_slice = (vis_slice * 255).astype(np.uint8)
                
                orig_img = Image.fromarray(vis_slice).convert("RGB")
                orig_resized = orig_img.resize((224, 224), Image.BILINEAR)
                
                # Prepare tensor for model
                img_array = np.array(orig_resized) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_array = (img_array - mean) / std
                img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Inference
                with torch.no_grad():
                    output = model(img_tensor)
                    probs = F.softmax(output, dim=1)
                    prediction = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
                
                # Create mask image
                mask_rgb = COLORS[prediction].astype(np.uint8)
                mask_image = Image.fromarray(mask_rgb)
                
                # Convert both to base64
                buf_orig = io.BytesIO()
                orig_img.resize((224,224)).save(buf_orig, format="PNG")
                original_slices.append(base64.b64encode(buf_orig.getvalue()).decode('utf-8'))
                
                buf_mask = io.BytesIO()
                mask_image.save(buf_mask, format="PNG")
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
