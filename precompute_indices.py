import pickle
from pathlib import Path
from typing import List, Dict

import torch
from decord import VideoReader
from tqdm import tqdm

NUM_FRAMES_TO_SAMPLE = 16
DATASET_ROOT = Path("CricketEC")
CACHE_FILE_NAME = "pixel_intensity_indices_32_frames.pkl"


def _calculate_likelihood_vectorized(
    frame1_chunks: torch.Tensor, frame2_chunks: torch.Tensor
) -> torch.Tensor:

    frame1_chunks = frame1_chunks.float()
    frame2_chunks = frame2_chunks.float()

    m_1 = frame1_chunks.mean(dim=(2, 3))
    m_2 = frame2_chunks.mean(dim=(2, 3))

    s_1 = frame1_chunks.var(dim=(2, 3))
    s_2 = frame2_chunks.var(dim=(2, 3))

    s_1[s_1 == 0] = 1
    s_2[s_2 == 0] = 1

    # calculate likelihood ratio components
    a = (s_1 + s_2) / 2
    b = ((m_1 - m_2) / 2) ** 2
    c = (a + b) ** 2
    d = s_1 * s_2

    # Final likelihood ratio per region
    # shape: (T-1, 16, 3)
    likelihood_ratio = c / d

    # average across the 3 color channels
    # shape: (T-1, 16)
    avg_likelihood_per_region = likelihood_ratio.mean(dim=2)

    # average across the 16 regions
    # shape: (T-1)
    total_frame_change_score = avg_likelihood_per_region.mean(dim=1)

    return total_frame_change_score


def calculate_indices_for_video(
    vr: VideoReader, num_frames_to_sample: int
) -> List[int]:
    nrows, ncols = 4, 4

    # Assuming 224x224 frames, (224*224) / 16 regions = 56x56 per region
    side_of_region = 56
    total_frames = len(vr)

    if total_frames < 2:
        indices = torch.linspace(
            0, max(0, total_frames - 1), num_frames_to_sample, dtype=torch.float32
        ).tolist()
        return [int(i) for i in indices]

    all_frames = torch.from_numpy(vr.get_batch(range(total_frames)).asnumpy())

    frames_prev = all_frames[:-1]
    frames_curr = all_frames[1:]

    def chunk_frames(frames: torch.Tensor) -> torch.Tensor:
        # Input shape: (T-1, 224, 224, 3)
        num_pairs = frames.shape[0]
        return (
            frames.reshape(num_pairs, nrows, side_of_region, ncols, side_of_region, 3)
            .permute(0, 1, 3, 2, 4, 5)  # (T-1, nrows, ncols, 56, 56, 3)
            .reshape(
                num_pairs, -1, side_of_region, side_of_region, 3
            )  # (T-1, 16, 56, 56, 3)
        )

    frame1_chunks = chunk_frames(frames_prev)
    frame2_chunks = chunk_frames(frames_curr)

    # 4. Run vectorized calculation
    # frame_scores shape: (T-1)
    frame_scores = _calculate_likelihood_vectorized(frame1_chunks, frame2_chunks)

    k = min(num_frames_to_sample - 1, len(frame_scores))
    if k <= 0:  # Handle videos with only 1 or 2 frames
        top_indices = torch.tensor([], dtype=torch.long)
    else:
        top_indices = torch.topk(frame_scores, k=k).indices + 1

    # Add frame 0 and sort the final indices
    final_indices = sorted(list(set([0] + top_indices.tolist())))

    # 6. Pad or truncate to ensure exactly num_frames_to_sample
    if len(final_indices) > num_frames_to_sample:
        final_indices = final_indices[:num_frames_to_sample]
    elif len(final_indices) < num_frames_to_sample:
        # Pad with the last frame index
        final_indices += [final_indices[-1]] * (
            num_frames_to_sample - len(final_indices)
        )

    return final_indices


def main():
    print(f"Dataset Root: {DATASET_ROOT.resolve()}")
    print(f"Target Frames per Video: {NUM_FRAMES_TO_SAMPLE}")
    print(f"Output Cache File: {CACHE_FILE_NAME}")

    # Find all videos in train, test, and val sets
    video_paths = list(DATASET_ROOT.glob("train/*/*.avi"))
    video_paths.extend(list(DATASET_ROOT.glob("test/*/*.avi")))
    video_paths.extend(list(DATASET_ROOT.glob("val/*/*.avi")))

    if not video_paths:
        print(f"\nError: No videos found at {DATASET_ROOT.resolve()}.")
        print("Please check that DATASET_ROOT is correct and your videos are in")
        print("subfolders like 'train/class_name/video.avi'")
        return

    print(f"\nFound {len(video_paths)} videos to process.")

    # This dictionary will store our results
    # Key: str (relative path like "train/cover_drive/video_001.avi")
    # Value: List[int] (the list of frame indices)
    index_map: Dict[str, List[int]] = {}

    for video_path in tqdm(video_paths, desc="Processing videos"):
        try:
            vr = VideoReader(str(video_path))
            indices = calculate_indices_for_video(vr, NUM_FRAMES_TO_SAMPLE)

            # Create a consistent, cross-platform key
            # e.g., "train/cover_drive/video_001.avi"
            relative_key = str(video_path.relative_to(DATASET_ROOT)).replace("\\", "/")
            index_map[relative_key] = indices

        except Exception as e:
            print(f"\nWarning: Failed to process {video_path}. Error: {e}")
            print("Skipping this video.")

    # Save the completed map
    cache_path = DATASET_ROOT / CACHE_FILE_NAME
    with open(cache_path, "wb") as f:
        pickle.dump(index_map, f)

    print("\n--- Success! ---")
    print(f"Processed {len(index_map)} videos.")
    print(f"Index map saved to: {cache_path.resolve()}")


if __name__ == "__main__":
    main()
