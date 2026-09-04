import torch
import model.dinov2

torch.cuda.reset_peak_memory_stats()
print("before load: allocated=%.1f MiB reserved=%.1f MiB" % (
    torch.cuda.memory_allocated() / 1024**2, torch.cuda.memory_reserved() / 1024**2))

dinov2_model, dinov2_transform = model.dinov2.load_dinov2_model(
    "cuda",
    "dinov2_vitl14",
    image_size=(630, 630),
    repo_or_dir="./dinov2",
    pretrained="./checkpoints/dinov2_vitl14_pretrain.pth",
)
torch.cuda.synchronize()

allocated = torch.cuda.memory_allocated() / 1024**2
reserved = torch.cuda.memory_reserved() / 1024**2
peak = torch.cuda.max_memory_allocated() / 1024**2
print(f"after load to cuda: allocated={allocated:.1f} MiB reserved={reserved:.1f} MiB peak={peak:.1f} MiB")

has_nan = any(torch.isnan(p).any().item() for p in list(dinov2_model.parameters())[:20])
has_inf = any(torch.isinf(p).any().item() for p in list(dinov2_model.parameters())[:20])
print(f"NaN in sampled params: {has_nan}, Inf in sampled params: {has_inf}")

print("DINOV2_LOAD_OK")
