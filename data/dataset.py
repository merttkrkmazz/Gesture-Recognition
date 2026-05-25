import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def load_split(csv_path, class_to_idx):
    """CSV'den sadece seçili class'lara ait (video_id, label) çiftlerini döndürür."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        label = str(row["label"]).strip()
        if label in class_to_idx:
            samples.append((str(int(row["video_id"])), class_to_idx[label]))
    return samples


def load_frames(video_dir, num_frames=37, img_size=112, as_tensor=True):
    """Klasördeki frame'leri sıraya göre okur, pad/truncate uygular."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    frame_files = sorted(
        [f for f in os.listdir(video_dir) if f.endswith((".jpg", ".png"))],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    frames = []
    for fname in frame_files[:num_frames]:
        img = Image.open(os.path.join(video_dir, fname)).convert("RGB")
        frames.append(transform(img))

    # Pad eksik frame'leri son frame ile doldur
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else torch.zeros(3, img_size, img_size))

    return torch.stack(frames)  # (T, C, H, W)


# ──────────────────────────────────────────────
# Model 1: Landmark dataset — yalnızca keypoint tensörü döndürür
# (extract_landmarks.py ile önceden işlenmiş .pt dosyaları gerekir)
# ──────────────────────────────────────────────
class LandmarkDataset(Dataset):
    def __init__(self, csv_path, landmark_dir, class_to_idx):
        self.samples = load_split(csv_path, class_to_idx)
        self.landmark_dir = landmark_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, label = self.samples[idx]
        path = os.path.join(self.landmark_dir, f"{video_id}.pt")
        landmarks = torch.load(path)   # (T, 63)
        return landmarks, label


# ──────────────────────────────────────────────
# Model 2: CNN-LSTM dataset — frame sequence döndürür
# ──────────────────────────────────────────────
class FrameSequenceDataset(Dataset):
    def __init__(self, csv_path, data_root, class_to_idx,
                 num_frames=37, img_size=224):
        self.samples = load_split(csv_path, class_to_idx)
        self.data_root = data_root
        self.num_frames = num_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, label = self.samples[idx]
        video_dir = os.path.join(self.data_root, video_id)
        frames = load_frames(video_dir, self.num_frames, self.img_size)
        return frames, label   # (T, C, H, W)


# ──────────────────────────────────────────────
# Model 3: 3D CNN dataset — (C, T, H, W) formatında döndürür
# ──────────────────────────────────────────────
class VideoDataset(Dataset):
    def __init__(self, csv_path, data_root, class_to_idx,
                 num_frames=37, img_size=112):
        self.samples = load_split(csv_path, class_to_idx)
        self.data_root = data_root
        self.num_frames = num_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, label = self.samples[idx]
        video_dir = os.path.join(self.data_root, video_id)
        frames = load_frames(video_dir, self.num_frames, self.img_size)
        # (T, C, H, W) -> (C, T, H, W)
        return frames.permute(1, 0, 2, 3), label
