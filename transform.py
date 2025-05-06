from PIL import Image
import numpy as np
import io
import torchvision.transforms as transforms

import random


class JpegCompression:
    def __init__(self, quality=75, probability=0.2):
        """
        :param quality: JPEG 压缩质量 (1-100)
        :param probability: 进行压缩的概率 (0-1)
        """
        self.quality = quality
        self.probability = probability

    def __call__(self, img):
        """对输入图像以一定概率进行 JPEG 压缩"""
        if random.random() < self.probability:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            return Image.open(buffer)
        return img


def get_augs(name="base", norm="imagenet", size=299):
    IMG_SIZE = size
    if norm == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif norm == "0.5":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]

    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # JpegCompression(quality=75, probability=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
