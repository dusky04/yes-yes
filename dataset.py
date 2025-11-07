from pathlib import Path
from typing import Optional, Tuple

import torch
from decord import VideoReader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import get_classes


class CricketEC(Dataset):
    def __init__(
        self,
        c,
        dir: Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.dataset_dir = dir
        # paths of all the videos present in the dataset - label / video.avi
        self.video_paths = list(dir.glob("*/*.avi"))
        self.class_names, self.class_to_idx = get_classes(dir)
        self.transform = transform
        self.config = c

    def __len__(self) -> int:
        return len(self.video_paths)

    def load_video_frames(self, idx: int) -> torch.Tensor:
        vr = VideoReader(str(self.video_paths[idx]))
        indices = torch.linspace(
            0, len(vr) - 1, self.config.NUM_FRAMES, dtype=torch.float32
        ).tolist()
        frames = torch.from_numpy(vr.get_batch(indices=indices).asnumpy()).permute(
            0, 3, 1, 2
        )
        if self.transform:
            return self.transform(frames)
        return frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.config.NUM_FRAMES > 32:
            raise Exception("CANT HANDLE 32 FRAMES")

        video = self.load_video_frames(idx)
        label = self.video_paths[idx].parent.name
        label_idx = self.class_to_idx[label]
        return video, label_idx


def get_dataloaders(
    config,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
) -> Tuple[DataLoader[CricketEC], DataLoader[CricketEC]]:
    train_dir = Path(config.DATASET_NAME) / "train"
    test_dir = Path(config.DATASET_NAME) / "test"

    train_dataset = CricketEC(c=config, dir=train_dir, transform=train_transform)
    test_dataset = CricketEC(c=config, dir=test_dir, transform=test_transform)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    return train_dataloader, test_dataloader


def get_val_dataloader(
    config, val_transform: transforms.Compose
) -> DataLoader[CricketEC]:
    val_dir = Path(config.DATASET_NAME) / "val"
    val_dataset = CricketEC(c=config, dir=val_dir, transform=val_transform)
    return DataLoader(
        val_dataset,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
