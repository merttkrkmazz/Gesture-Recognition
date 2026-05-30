import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_split(csv_path, class_to_idx):
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        label = str(row["label"]).strip()
        if label in class_to_idx:
            samples.append((str(int(row["video_id"])), class_to_idx[label]))
    return samples


def load_frames(video_dir, num_frames=37, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    frame_files = sorted(
        [f for f in os.listdir(video_dir) if f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    frames = []
    for fname in frame_files[:num_frames]:
        img = Image.open(os.path.join(video_dir, fname)).convert("RGB")
        frames.append(transform(img))
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else torch.zeros(3, img_size, img_size))
    return torch.stack(frames)  # (T, C, H, W)


class GestureDataset(Dataset):
    def __init__(self, csv_path, data_root, class_to_idx,
                 num_frames=37, img_size=224):
        self.samples    = load_split(csv_path, class_to_idx)
        self.data_root  = data_root
        self.num_frames = num_frames
        self.img_size   = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, label = self.samples[idx]
        video_dir = os.path.join(self.data_root, video_id)
        frames = load_frames(video_dir, self.num_frames, self.img_size)
        return frames, label  # (T, C, H, W)
