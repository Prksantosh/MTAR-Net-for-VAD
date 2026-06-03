import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset


class ShanghaiTech(Dataset):

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