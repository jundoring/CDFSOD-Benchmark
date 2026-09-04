import os
import torch
from mmengine import Config
import chatrex.upn.transforms.transform as T  # noqa
from chatrex.upn import build_architecture
import chatrex.upn.inference_wrapper as iw

CKPT = "./checkpoints/upn_large.pth"

torch.cuda.reset_peak_memory_stats()
print("before load: allocated=%.1f MiB reserved=%.1f MiB" % (
    torch.cuda.memory_allocated() / 1024**2, torch.cuda.memory_reserved() / 1024**2))

config_path = os.path.join(os.path.dirname(os.path.abspath(iw.__file__)), "configs/upn_large.py")
model_cfg = Config.fromfile(config_path).model
model = build_architecture(model_cfg)
checkpoint = torch.load(CKPT, map_location="cpu")
result = model.load_state_dict(checkpoint["model"], strict=False)
print(f"checkpoint={CKPT}")
print(f"missing_keys: {len(result.missing_keys)}")
print(f"unexpected_keys: {len(result.unexpected_keys)}")
if result.missing_keys:
    print("  sample missing:", result.missing_keys[:5])
if result.unexpected_keys:
    print("  sample unexpected:", result.unexpected_keys[:5])

model.eval()
model.to("cuda")
torch.cuda.synchronize()

allocated = torch.cuda.memory_allocated() / 1024**2
reserved = torch.cuda.memory_reserved() / 1024**2
peak = torch.cuda.max_memory_allocated() / 1024**2
print(f"after load to cuda: allocated={allocated:.1f} MiB reserved={reserved:.1f} MiB peak={peak:.1f} MiB")

# NaN/Inf check on a sample of params
has_nan = any(torch.isnan(p).any().item() for p in list(model.parameters())[:20])
has_inf = any(torch.isinf(p).any().item() for p in list(model.parameters())[:20])
print(f"NaN in sampled params: {has_nan}, Inf in sampled params: {has_inf}")

print("UPN_LOAD_OK")
