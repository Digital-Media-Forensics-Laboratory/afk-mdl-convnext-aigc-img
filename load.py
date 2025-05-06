from torch.utils.data import DataLoader
import torch
from convext import ConvNext
from transform import get_augs
from torch.optim.lr_scheduler import StepLR
import os

model_path = "./epoch_0_batch_3000_acc_98.845_auc_99.958.pth"

if __name__ == "__main__":



    model = ConvNext(num_classes=2).to("cuda")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model"])

    # val_augs = get_augs(name="None", norm="0.5", size=512)

    # res = model.predict("images/fake_image_1.jpeg")
    # print(res)

    files = os.listdir("real_images")
    for f in files:
        resource = os.path.join("real_images", f)
        print(resource)
        res = model.predict(resource)
        print(res)
    
