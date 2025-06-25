import torch
print("✅ CUDA Available:", torch.cuda.is_available())
print("🖥️ GPU:", torch.cuda.get_device_name(0))
