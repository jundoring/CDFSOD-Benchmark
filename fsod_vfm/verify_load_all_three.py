import torch
from chatrex.upn import UPNWrapper
import model.dinov2
import model.sam2

torch.cuda.reset_peak_memory_stats()

print("Loading UPN...")
upn = UPNWrapper("./checkpoints/upn_large.pth")
torch.cuda.synchronize()
print("after UPN: allocated=%.1f MiB reserved=%.1f MiB" % (
    torch.cuda.memory_allocated() / 1024**2, torch.cuda.memory_reserved() / 1024**2))

print("Loading DINOv2...")
dinov2_model, dinov2_transform = model.dinov2.load_dinov2_model(
    "cuda", "dinov2_vitl14", image_size=(630, 630),
    repo_or_dir="./dinov2", pretrained="./checkpoints/dinov2_vitl14_pretrain.pth",
)
torch.cuda.synchronize()
print("after UPN+DINOv2: allocated=%.1f MiB reserved=%.1f MiB" % (
    torch.cuda.memory_allocated() / 1024**2, torch.cuda.memory_reserved() / 1024**2))

print("Loading SAM2...")
sam2_model, sam2_predictor, sam2_mask_generator = model.sam2.load_sam2_components(
    model_type="large", device="cuda", points_per_side=32,
)
torch.cuda.synchronize()

allocated = torch.cuda.memory_allocated() / 1024**2
reserved = torch.cuda.memory_reserved() / 1024**2
peak = torch.cuda.max_memory_allocated() / 1024**2
print(f"after all three: allocated={allocated:.1f} MiB reserved={reserved:.1f} MiB peak={peak:.1f} MiB")

total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
print(f"GPU total memory: {total_gpu_mem:.1f} MiB")
print(f"headroom remaining (total - reserved): {total_gpu_mem - reserved:.1f} MiB")

print("ALL_THREE_LOAD_OK")
