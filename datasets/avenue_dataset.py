import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class Avenuedataset(Dataset):

    def __init__(self, root_dir, seq_len=4, transform=None):

        self.seq_len = seq_len
        self.transform = transform
        self.samples = []

        videos = sorted(os.listdir(root_dir))

        for vid in videos:

            frames = sorted(
                glob.glob(os.path.join(root_dir, vid, "*.jpg"))
            )

            for i in range(len(frames) - seq_len):

                input_seq = frames[i:i+seq_len]
                target = frames[i+seq_len]

                self.samples.append((input_seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        seq_paths, target_path = self.samples[idx]

        frames = []

        for p in seq_paths:

            img = Image.open(p).convert("RGB")

            if self.transform:
                img = self.transform(img)

            frames.append(img)

        frames = torch.stack(frames)

        target = Image.open(target_path).convert("RGB")

        if self.transform:
            target = self.transform(target)

        return frames, target
    



class AvenueValDataset(Dataset):
    """
    Label-aware validation dataset for sequence-to-frame prediction.

    Expected folder structure:
        root_dir/
            video_01/
                0001.jpg
                0002.jpg
                ...
            video_02/
                0001.jpg
                0002.jpg
                ...

    Labels are built from anomaly_ranges_dict:
        {
            "video_01": [(start1, end1), (start2, end2)],
            "video_02": [(start3, end3)],
            ...
        }

    By default, frame ranges are assumed 1-based and inclusive.
    The label returned for each sample corresponds to the target frame.
    """

    def __init__(
        self,
        root_dir,
        seq_len=4,
        transform=None,
        anomaly_ranges_dict=None,
        one_based=True,
        frame_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    ):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.one_based = one_based
        self.anomaly_ranges_dict = anomaly_ranges_dict or {}
        self.frame_extensions = tuple(ext.lower() for ext in frame_extensions)

        self.samples = []

        video_folders = []
        for item in sorted(os.listdir(root_dir)):
            vid_path = os.path.join(root_dir, item)
            if os.path.isdir(vid_path):
                video_folders.append(vid_path)

        print(f"Total validation video folders found: {len(video_folders)}")

        for vid_path in video_folders:
            video_name = os.path.basename(vid_path)

            frames = []
            for ext in self.frame_extensions:
                frames.extend(glob.glob(os.path.join(vid_path, f"*{ext}")))
            frames = sorted(frames)

            if len(frames) <= seq_len:
                continue

            # Build per-frame labels for this video
            frame_labels = self._build_video_labels(
                num_frames=len(frames),
                anomaly_ranges=self.anomaly_ranges_dict.get(video_name, [])
            )

            # Create sequence samples
            for i in range(len(frames) - seq_len):
                input_seq = frames[i:i + seq_len]
                target = frames[i + seq_len]

                # target frame label
                target_label = int(frame_labels[i + seq_len])

                self.samples.append((input_seq, target, target_label, video_name, i + seq_len))

        print(f"Total validation samples: {len(self.samples)}")

    def _build_video_labels(self, num_frames, anomaly_ranges):
        labels = np.zeros(num_frames, dtype=np.int32)

        for start, end in anomaly_ranges:
            if self.one_based:
                start = start - 1
                end = end - 1

            start = max(0, start)
            end = min(num_frames - 1, end)

            if end >= start:
                labels[start:end + 1] = 1

        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_paths, target_path, label, video_name, target_index = self.samples[idx]

        frames = []
        for p in seq_paths:
            img = Image.open(p).convert("RGB")

            if self.transform is not None:
                img = self.transform(img)
            else:
                img = torch.from_numpy(
                    np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
                )

            frames.append(img)

        frames = torch.stack(frames)  # (T, C, H, W)

        target = Image.open(target_path).convert("RGB")
        if self.transform is not None:
            target = self.transform(target)
        else:
            target = torch.from_numpy(
                np.array(target, dtype=np.float32).transpose(2, 0, 1) / 255.0
            )

        label = torch.tensor(label, dtype=torch.long)

        return frames, target, label    