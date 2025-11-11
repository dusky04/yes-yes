from dataclasses import dataclass

import torch
from torch import nn
from pathlib import Path
from utils import setup_and_download_dataset
from dataset import get_dataloaders, FrameSampling
from models.cnn import resnet34_lstm_model
from torchvision import transforms
from train import train_model
from torch.optim.lr_scheduler import ReduceLROnPlateau


@dataclass
class C:
    LR = 3e-3
    DATASET_NAME = "CricketEC-train-test-val"
    NUM_CLASSES = 15
    NUM_FRAMES = 16
    BATCH_SIZE = 16
    LSTM_HIDDEN_DIM = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    FC_DROPOUT = 0.4
    TRAIN_SIZE = 0.8
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 10
    NUM_EPOCHS = 40
    WEIGHT_DECAY = 3e-3


if __name__ == "__main__":
    # setup the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running Device:", device)

    # setup the dataset
    DATASET_NAME = "CricketEC"
    CRICKET_EC_URL = "https://drive.google.com/file/d/1QRM360a5HvRKvPF3k7vOT1PS0suxioJd/view?usp=sharing"
    setup_and_download_dataset(
        DATASET_NAME, url=CRICKET_EC_URL, download_dir=Path("zipped_data")
    )

    # setup config (if needed)
    c = C()

    # setup transforms
    train_transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.4),
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.15, 0.15))], p=0.4
            ),
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.3),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=(7, 7))], p=0.5
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # setup dataloaders
    train_dataloader, test_dataloader = get_dataloaders(
        c,
        train_transform=train_transform,
        test_transform=test_transform,
        sampling=FrameSampling.PIXEL_INTENSITY,
    )

    # setup model
    model = resnet34_lstm_model(c).to(device)
    # model.compile()

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(
        [
            {"params": model.feature_extractor.parameters(), "lr": 1e-5},
            {"params": model.lstm.parameters()},
            {"params": model.linear.parameters()},
            {"params": model.dropout.parameters()},
        ],
        lr=c.LR,
        weight_decay=c.WEIGHT_DECAY,
    )

    # lr-scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    # train
    train_model(
        c=c,
        exp_name="resnet34-weights",
        weights_dir=Path("weights"),
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
