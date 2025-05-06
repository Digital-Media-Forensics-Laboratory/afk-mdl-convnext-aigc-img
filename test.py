from torch.utils.data import DataLoader
import torch
from convext import ConvNext
from transform import get_augs
from torch.optim.lr_scheduler import StepLR
import os

model_path = "./epoch_2_auc_99.958.pth"

if __name__ == "__main__":
    model = ConvNext(num_classes=2).to("cuda")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model"])
    res = model.predict("images/stable_diffusion.jpeg")
    print(res)
