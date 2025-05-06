import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from transform import get_augs


class ConvNext(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNext, self).__init__()
        self.model = timm.create_model("convnext_base", pretrained=False)
        # self.model = timm.create_model("convnext_base", pretrained=True)
        self.fc = nn.Linear(1000, num_classes)
        self.val_augs = get_augs(name="None", norm="0.5", size=512)

    def forward(self, x):
        x = self.model(x)
        y = self.fc.forward(x)
        return [x, y]

    def predict(self, image: str):
        im = Image.open(image).convert("RGB")
        im = self.val_augs(im)
        t = im.unsqueeze(0).to("cuda")

        x = self.model(t)
        y = self.fc.forward(x)
        return torch.softmax(y, dim=1).cpu().detach().numpy()[0]


if __name__ == "__main__":
    model = ConvNext()
    image = torch.rand(64, 3, 224, 224)
    feature, data = model(image)
    print(data.shape)
    print(data)
