from torchvision.models.video import r3d_18
from torch import nn


def video_resnet(c):
    resnet = r3d_18(weights="DEFAULT")
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Linear(in_features=512, out_features=c.NUM_CLASSES)
    return resnet
