import torch
import model.sam2

torch.cuda.reset_peak_memory_stats()
print("before load: allocated=%.1f MiB reserved=%.1f MiB" % (
    torch.cuda.memory_allocated() / 1024**2, torch.cuda.memory_reserved() / 1024**2))

sam2_model, sam2_predictor, sam2_mask_generator = model.sam2.load_sam2_components(
    model_type="large",
    device="cuda",
    points_per_side=32,
)
torch.cuda.synchronize()

allocated = torch.cuda.memory_allocated() / 1024**2
reserved = torch.cuda.memory_reserved() / 1024**2
peak = torch.cuda.max_memory_allocated() / 1024**2
print(f"after load to cuda: allocated={allocated:.1f} MiB reserved={reserved:.1f} MiB peak={peak:.1f} MiB")

has_nan = any(torch.isnan(p).any().item() for p in list(sam2_model.parameters())[:20])
has_inf = any(torch.isinf(p).any().item() for p in list(sam2_model.parameters())[:20])
print(f"NaN in sampled params: {has_nan}, Inf in sampled params: {has_inf}")

print("SAM2_LOAD_OK")
