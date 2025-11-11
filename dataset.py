from pathlib import Path
from typing import Optional, Tuple, List

import pickle
import torch
from decord import VideoReader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils import get_classes
from enum import Enum


class FrameSampling(Enum):
    UNIFORM = 0
    JITTERED = 1
    PIXEL_INTENSITY = 2


# def _calculate_likelihood(
#     frame1_chunks: torch.Tensor, frame2_chunks: torch.Tensor
# ) -> torch.Tensor:
#     frame1_chunks = frame1_chunks.float()
#     frame2_chunks = frame2_chunks.float()

#     # m_i and m_i+1 represents
#     # mean intensity values for a given colour channel for the
#     # region in two consecutive frames
#     m_1 = frame1_chunks.mean(dim=(1, 2))
#     m_2 = frame2_chunks.mean(dim=(1, 2))

#     # s_i and s_i+1 represents
#     # corresponding variance values for a given colour channel for the
#     # region in two consecutive frames
#     s_1 = frame1_chunks.var(dim=(1, 2))
#     s_2 = frame2_chunks.var(dim=(1, 2))

#     # handle zero variance
#     # "If s_i = 0 or s_{i+1} = 0 then set s_i = s_{i+1} = 1"
#     # We set any 0 variance to 1 to avoid division by zero.
#     s_1[s_1 == 0] = 1
#     s_2[s_2 == 0] = 1

#     # calculate likelihood ratio components
#     a = (s_1 + s_2) / 2
#     b = ((m_1 - m_2) / 2) ** 2
#     c = (a + b) ** 2
#     d = s_1 * s_2

#     # Final likelihood ratio per region
#     likelihood_ratio = c / d

#     # average across the 3 color channels
#     avg_likelihood_per_region = likelihood_ratio.mean(dim=1)

#     return avg_likelihood_per_region


# def _pixel_intensity_sampling(c, vr: VideoReader) -> List[int]:
#     # this function handles frames in `HWC` format
#     # for our use case it would become (224, 224, 3) which is the numpy expected format
#     # however, since we use torch we need our frames in `CHW` format
#     # convert everything to tensors and then permute in the `CHW` format

#     nrows, ncols = 4, 4
#     side_of_region = 56  # (width * height)  / num_of_regions -> (224 * 224) / 16

#     frame_prev = torch.from_numpy(vr[0].asnumpy())
#     total_frames = len(vr)

#     frame_scores = []

#     for i in range(1, total_frames):
#         # shape -> (224, 224, 3) -> (HWC)
#         frame_curr = torch.from_numpy(vr[i].asnumpy())

#         frame1_chunks = (
#             frame_prev.reshape(nrows, side_of_region, ncols, side_of_region, 3)
#             .permute(0, 2, 1, 3, 4)
#             .reshape(-1, side_of_region, side_of_region, 3)
#         )

#         frame2_chunks = (
#             frame_curr.reshape(nrows, side_of_region, ncols, side_of_region, 3)
#             .permute(0, 2, 1, 3, 4)
#             .reshape(-1, side_of_region, side_of_region, 3)
#         )

#         likelihood_scores_per_region = _calculate_likelihood(
#             frame1_chunks, frame2_chunks
#         )

#         total_frame_change_score = likelihood_scores_per_region.mean()

#         frame_scores.append((total_frame_change_score, i))

#         frame_prev = frame_curr

#     sorted_frames = sorted(frame_scores, key=lambda x: x[0], reverse=True)

#     # Get the indices of the top N-1 frames with the most change
#     # We subtract 1 because we will always add frame 0 by default
#     top_indices = [idx for score, idx in sorted_frames[: c.NUM_FRAMES - 1]]

#     # Add frame 0 and sort the final indices
#     final_indices = sorted(list(set([0] + top_indices)))

#     # Ensure exactly NUM_FRAMES frames
#     if len(final_indices) > c.NUM_FRAMES:
#         final_indices = final_indices[: c.NUM_FRAMES]
#     elif len(final_indices) < c.NUM_FRAMES:
#         final_indices += [final_indices[-1]] * (c.NUM_FRAMES - len(final_indices))

#     return final_indices


# def _pixel_intensity_sampling(c, vr: VideoReader) -> List[int]:
#     nrows, ncols = 4, 4
#     side_of_region = 56
#     total_frames = len(vr)

#     if total_frames < 2:
#         # Not enough frames to compare, return uniform
#         indices = torch.linspace(
#             0, total_frames - 1, c.NUM_FRAMES, dtype=torch.float32
#         ).tolist()
#         return [int(i) for i in indices] # Ensure integer indices

#     # 1. Batch read all frames.
#     # Shape -> (T, H, W, 3)
#     all_frames = torch.from_numpy(vr.get_batch(range(total_frames)).asnumpy())

#     # 2. Create (T-1) pairs of frames
#     # shape -> (T-1, H, W, 3)
#     frames_prev = all_frames[:-1]
#     frames_curr = all_frames[1:]

#     # 3. Vectorize the chunking operation
#     def chunk_frames(frames: torch.Tensor) -> torch.Tensor:
#         # Input shape: (T-1, 224, 224, 3)
#         num_pairs = frames.shape[0]
#         return (
#             frames.reshape(num_pairs, nrows, side_of_region, ncols, side_of_region, 3)
#             .permute(0, 1, 3, 2, 4, 5) # (T-1, nrows, ncols, 56, 56, 3)
#             .reshape(num_pairs, -1, side_of_region, side_of_region, 3) # (T-1, 16, 56, 56, 3)
#         )

#     frame1_chunks = chunk_frames(frames_prev)
#     frame2_chunks = chunk_frames(frames_curr)

#     # 4. Run vectorized calculation
#     # frame_scores shape: (T-1)
#     frame_scores = _calculate_likelihood(frame1_chunks, frame2_chunks)

#     # 5. Get top indices
#     # We add 1 to the index because the score at index 0 corresponds to the change
#     # between frame 0 and frame 1 (so it belongs to frame 1).
#     top_indices = torch.topk(frame_scores, k=min(c.NUM_FRAMES - 1, len(frame_scores))).indices + 1

#     # Add frame 0 and sort the final indices
#     final_indices = sorted(list(set([0] + top_indices.tolist())))

#     # 6. Pad or truncate
#     if len(final_indices) > c.NUM_FRAMES:
#         final_indices = final_indices[: c.NUM_FRAMES]
#     elif len(final_indices) < c.NUM_FRAMES:
#         final_indices += [final_indices[-1]] * (c.NUM_FRAMES - len(final_indices))

#     return final_indices


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

        self.index_map = None
        cache_path = Path(
            f"CricketEC/pixel_intensity_indices_{self.config.NUM_FRAMES}_frames.pkl"
        )
        if self.sampling == FrameSampling.PIXEL_INTENSITY:
            print("Loaded indices:", str(cache_path))
            with open(cache_path, "rb") as f:
                self.index_map = pickle.load(f)

    def __len__(self) -> int:
        return len(self.video_paths)

    def load_video_frames(self, idx: int) -> torch.Tensor:
        vr = VideoReader(str(self.video_paths[idx]))
        video_path = self.video_paths[idx]

        indices = []
        match self.sampling:
            case FrameSampling.UNIFORM:
                indices = torch.linspace(
                    0, len(vr) - 1, self.config.NUM_FRAMES, dtype=torch.float32
                ).tolist()
            case FrameSampling.JITTERED:
                raise NotImplementedError
            case FrameSampling.PIXEL_INTENSITY:
                if self.index_map is None:
                    raise FileNotFoundError("INDEX MAP NOT LOADING")

                key = str(video_path.relative_to(self.dataset_dir.parent)).replace(
                    "\\", "/"
                )

                if key not in self.index_map:
                    raise KeyError(
                        f"Video {key} not found in pre-computed index map.")

                indices = self.index_map[key]

                if len(indices) != self.config.NUM_FRAMES:

                    if len(indices) > self.config.NUM_FRAMES:
                        indices = indices[: self.config.NUM_FRAMES]
                    elif len(indices) < self.config.NUM_FRAMES:
                        indices += [indices[-1]] * (
                            self.config.NUM_FRAMES - len(indices)
                        )

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
