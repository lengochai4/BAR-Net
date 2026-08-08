"""
Kiểm tra forward pass của Final Version (Fourier + CPEA).
Chạy trên server: python verify_model.py
"""
import torch
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

print("=" * 60)
print("Verifying Final Version: Fourier + CPEA")
print("=" * 60)

model = SwinTransformerSys(
    img_size=224,
    num_classes=9,
    embed_dim=96,
    depths=[2, 2, 2, 2],
    depths_decoder=[1, 2, 2, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
).cuda()

x = torch.randn(2, 3, 224, 224).cuda()

with torch.no_grad():
    out = model(x)

print(f"\n✅ Forward pass OK!")
print(f"   Input  shape : {x.shape}")
print(f"   Output shape : {out.shape}")

# Count parameters
total = sum(p.numel() for p in model.parameters())
skip_fourier = sum(p.numel() for p in model.skip_filters.parameters())
skip_cpea    = sum(p.numel() for p in model.cpea_blocks.parameters())
print(f"\n📊 Parameter count:")
print(f"   Total model    : {total:,}")
print(f"   FreqFilter only: {skip_fourier:,}  ({100*skip_fourier/total:.2f}%)")
print(f"   CPEA only      : {skip_cpea:,}  ({100*skip_cpea/total:.2f}%)")
print(f"   Combined overhead: {100*(skip_fourier+skip_cpea)/total:.2f}%")
print("\n✅ All checks passed. Ready to train!")
