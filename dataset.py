from pathlib import Path
from typing import Optional, Tuple, List

import torch
from decord import VideoReader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from utils import get_classes
from enum import Enum


class FrameSampling(Enum):
    UNIFORM = 0
    JITTERED = 1
    PIXEL_INTENSITY = 2


def _calculate_likelihood(
    frame1_chunks: torch.Tensor, frame2_chunks: torch.Tensor
) -> torch.Tensor:
    # m_i and m_i+1 represents
    # mean intensity values for a given colour channel for the
    # region in two consecutive frames
    m_1 = frame1_chunks.mean(axis=(1, 2))
    m_2 = frame2_chunks.mean(axis=(1, 2))

    # s_i and s_i+1 represents
    # corresponding variance values for a given colour channel for the
    # region in two consecutive frames
    s_1 = frame1_chunks.var(axis=(1, 2))
    s_2 = frame2_chunks.var(axis=(1, 2))

    # handle zero variance
    # "If s_i = 0 or s_{i+1} = 0 then set s_i = s_{i+1} = 1"
    # We set any 0 variance to 1 to avoid division by zero.
    s_1[s_1 == 0] = 1
    s_2[s_2 == 0] = 1

    # calculate likelihood ratio components
    a = (s_1 + s_2) / 2
    b = ((m_1 - m_2) / 2) ** 2
    c = (a + b) ** 2
    d = s_1 * s_2

    # Final likelihood ratio per region
    likelihood_ratio = c / d

    # average across the 3 color channels
    avg_likelihood_per_region = likelihood_ratio.mean(axis=1)

    return avg_likelihood_per_region


def _pixel_intensity_sampling(c, vr: VideoReader) -> List[int]:
    # this function handles frames in `HWC` format
    # for our use case it would become (224, 224, 3) which is the numpy expected format
    # however, since we use torch we need our frames in `CHW` format
    # convert everything to tensors and then permute in the `CHW` format

    nrows, ncols = 4, 4
    side_of_region = 56  # (width * height)  / num_of_regions -> (224 * 224) / 16

    frame_prev = vr[0]
    total_frames = len(vr)

    frame_scores = []

    for i in range(1, total_frames):
        # shape -> (224, 224, 3) -> (HWC)
        frame_curr = torch.from_numpy(vr[i].asnumpy())

        frame1_chunks = (
            frame_prev.reshape(nrows, side_of_region, ncols, side_of_region, 3)
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, side_of_region, side_of_region, 3)
        )

        frame2_chunks = (
            frame_curr.reshape(nrows, side_of_region, ncols, side_of_region, 3)
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, side_of_region, side_of_region, 3)
        )

        likelihood_scores_per_region = _calculate_likelihood(
            frame1_chunks, frame2_chunks
        )

        total_frame_change_score = likelihood_scores_per_region.mean()

        frame_scores.append((total_frame_change_score, i))

        frame_prev = frame_curr

    sorted_frames = sorted(frame_scores, key=lambda x: x[0], reverse=True)

    # Get the indices of the top N-1 frames with the most change
    # We subtract 1 because we will always add frame 0 by default
    top_indices = [idx for score, idx in sorted_frames[: c.NUM_FRAMES - 1]]

    # Add frame 0 and sort the final indices
    final_indices = sorted(list(set([0] + top_indices)))

    return final_indices


class CricketEC(Dataset):
    def __init__(
        self,
        c,
        dir: Path,
        transform: Optional[transforms.Compose] = None,
        sampling: FrameSampling = FrameSampling.UNIFORM,
    ) -> None:
        self.dataset_dir = dir
        # paths of all the videos present in the dataset - label / video.avi
        self.video_paths = list(dir.glob("*/*.avi"))
        self.class_names, self.class_to_idx = get_classes(dir)
        self.transform = transform
        self.sampling = sampling
        self.config = c

    def __len__(self) -> int:
        return len(self.video_paths)

    def load_video_frames(self, idx: int) -> torch.Tensor:
        vr = VideoReader(str(self.video_paths[idx]))

        indices = []
        match self.sampling:
            case FrameSampling.UNIFORM:
                indices = torch.linspace(
                    0, len(vr) - 1, self.config.NUM_FRAMES, dtype=torch.float32
                ).tolist()
            case FrameSampling.JITTERED:
                raise NotImplementedError
            case FrameSampling.PIXEL_INTENSITY:
                indices = _pixel_intensity_sampling(self.config, vr)

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
    sampling: FrameSampling = FrameSampling.UNIFORM,
) -> Tuple[DataLoader[CricketEC], DataLoader[CricketEC]]:
    train_dir = Path(config.DATASET_NAME) / "train"
    test_dir = Path(config.DATASET_NAME) / "test"

    train_dataset = CricketEC(
        c=config, dir=train_dir, transform=train_transform, sampling=sampling
    )
    test_dataset = CricketEC(
        c=config, dir=test_dir, transform=test_transform, sampling=sampling
    )

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
