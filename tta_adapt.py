import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
from monai.data import Dataset, DataLoader
from monai.transforms import EnsureChannelFirst, ScaleIntensityRangePercentiles, ToTensor
from omegaconf import OmegaConf

cfg = OmegaConf.load("vesselfm/seg/configs/inference.yaml")

from vesselfm.seg.model import load_model
model = load_model(cfg)
model.cuda()

from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(repo_id="bwittmann/vesselFM", filename="vesselFM_all.pt")
model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
print("Модель загружена.")

for param in model.parameters():
    param.requires_grad = False

unfrozen_params = []
for name, module in model.named_modules():
    if 'norm' in name.lower() or 'bn' in name.lower():
        for param in module.parameters():
            param.requires_grad = True
            unfrozen_params.append(param)
print(f"Разморожено параметров (Norm): {len(unfrozen_params)}")

optimizer = torch.optim.Adam(unfrozen_params, lr=1e-4)

data_dir = "D:/vesselFM/my_data"
output_dir = "D:/vesselFM/my_output_tta"
os.makedirs(output_dir, exist_ok=True)

image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nii.gz')]

transforms = [
    EnsureChannelFirst(channel_dim="no_channel"),
    ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=1, clip=True),
    ToTensor(device='cuda')
]
dataset = Dataset(data=[{"image": img} for img in image_files], transform=transforms)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model.train()
entropy_threshold = 0.5

for i, batch_data in enumerate(loader):
    inputs = batch_data["image"].cuda()
    optimizer.zero_grad()
    
    outputs = model(inputs)
    probs = torch.sigmoid(outputs)
    
    entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8)).mean()
    
    if entropy.item() < entropy_threshold:
        loss = entropy
        loss.backward()
        optimizer.step()
        print(f"Файл {i+1}/{len(loader)}: адаптация выполнена (энтропия {entropy.item():.4f})")
    else:
        print(f"Файл {i+1}/{len(loader)}: пропущено (высокая энтропия {entropy.item():.4f})")
    
    # Сохранение результата
    pred_mask = (probs > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    img = sitk.GetImageFromArray(pred_mask)
    out_name = os.path.basename(image_files[i]).replace('.nii.gz', '_tta_pred.nii.gz')
    sitk.WriteImage(img, os.path.join(output_dir, out_name))

print("TTA завершена. Результаты в", output_dir)