# ================================================================
# CS466 — Gesture Recognition: CNN-LSTM Model
# Kaggle notebook'a hücre hücre yapıştır (her # HÜCRE = yeni hücre)
# ================================================================

# ── HÜCRE 1: Path kontrol ────────────────────────────────────────
import os

# Dataset adını ne koyduğuna göre değişebilir, bunu çalıştır önce
for root, dirs, files in os.walk("/kaggle/input"):
    level = root.replace("/kaggle/input", "").count(os.sep)
    if level < 3:
        print(root)

# ── HÜCRE 2: Config ──────────────────────────────────────────────
import os, torch, random, time
import numpy as np

GESTURE_CLASSES = [
    "Doing other things",
    "Drumming Fingers",
    "No gesture",
    "Pulling Hand In",
    "Pulling Two Fingers In",
    "Pushing Hand Away",
    "Pushing Two Fingers Away",
    "Rolling Hand Backward",
    "Rolling Hand Forward",
    "Shaking Hand",
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(GESTURE_CLASSES)}
NUM_CLASSES  = len(GESTURE_CLASSES)

NUM_FRAMES   = 37
IMG_SIZE     = 112      # MobileNetV2 için 112 yeterli, daha hızlı
BATCH_SIZE   = 16
NUM_EPOCHS   = 15
LR           = 1e-4
WEIGHT_DECAY = 1e-4
SEED         = 42

# !! Hücre 1 çıktısına göre bu path'i güncelle
BASE       = "/kaggle/input/jester-subset/Jester20bn_subset"
TRAIN_ROOT = f"{BASE}/train_subset"
VAL_ROOT   = f"{BASE}/val_subset"
TRAIN_CSV  = f"{BASE}/train_subset/train_subset.csv"
VAL_CSV    = f"{BASE}/val_subset/val_subset.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
print(f"Device: {DEVICE} | Classes: {NUM_CLASSES}")
print(f"Train CSV exists: {os.path.exists(TRAIN_CSV)}")

# ── HÜCRE 3: Dataset ─────────────────────────────────────────────
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_split(csv_path):
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        label = str(row["label"]).strip()
        if label in CLASS_TO_IDX:
            samples.append((str(int(row["video_id"])), CLASS_TO_IDX[label]))
    return samples


def load_frames(video_dir, num_frames, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    files = sorted(
        [f for f in os.listdir(video_dir) if f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    frames = []
    for fname in files[:num_frames]:
        img = Image.open(os.path.join(video_dir, fname)).convert("RGB")
        frames.append(transform(img))
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else torch.zeros(3, img_size, img_size))
    return torch.stack(frames)  # (T, C, H, W)


class GestureDataset(Dataset):
    def __init__(self, csv_path, data_root):
        self.samples   = load_split(csv_path)
        self.data_root = data_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, label = self.samples[idx]
        frames = load_frames(
            os.path.join(self.data_root, video_id),
            NUM_FRAMES, IMG_SIZE
        )
        return frames, label  # (T, C, H, W)


train_dataset = GestureDataset(TRAIN_CSV, TRAIN_ROOT)
val_dataset   = GestureDataset(VAL_CSV,   VAL_ROOT)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} sample")

# class dağılımı
from collections import Counter
train_labels = [s[1] for s in train_dataset.samples]
print("\nClass dağılımı:")
for idx, count in sorted(Counter(train_labels).items()):
    print(f"  {GESTURE_CLASSES[idx]:<30} {count}")

# ── HÜCRE 4: CNN-LSTM Model ──────────────────────────────────────
import torch.nn as nn
from torchvision import models


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_dim=256,
                 num_layers=2, dropout=0.3):
        super().__init__()

        # MobileNetV2 backbone — son classification layer'ı kaldır
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT)
        self.feature_dim = backbone.last_channel  # 1280
        backbone.classifier = nn.Identity()
        self.cnn = backbone

        # CNN'i freeze et — sadece LSTM + head train olur (daha hızlı)
        for p in self.cnn.features[:-3].parameters():
            p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)                  # (B*T, 1280)
        feats = feats.view(B, T, -1)         # (B, T, 1280)
        out, _ = self.lstm(feats)            # (B, T, hidden_dim)
        out = out[:, -1, :]                  # son timestep
        return self.head(out)


model = CNNLSTM().to(DEVICE)
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params    : {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

# ── HÜCRE 5: Training ────────────────────────────────────────────
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


criterion = nn.CrossEntropyLoss()
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0

print(f"{'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>9} {'ValLoss':>8} {'ValAcc':>8} {'Time':>6}")
print("-" * 55)

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
    vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, criterion)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)

    flag = " ✓" if vl_acc > best_val_acc else ""
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), "/kaggle/working/cnn_lstm_best.pth")

    print(f"{epoch:>5} {tr_loss:>10.4f} {tr_acc:>9.4f} "
          f"{vl_loss:>8.4f} {vl_acc:>8.4f} {time.time()-t0:>5.0f}s{flag}")

print(f"\nBest Val Accuracy: {best_val_acc:.4f}")

# ── HÜCRE 6: Evaluation ──────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                              f1_score, accuracy_score)

# En iyi modeli yükle
model.load_state_dict(torch.load("/kaggle/working/cnn_lstm_best.pth"))

_, test_acc, preds, labels = eval_epoch(model, val_loader, criterion)
f1 = f1_score(labels, preds, average="macro")

print("=" * 50)
print(f"Top-1 Accuracy : {test_acc:.4f}")
print(f"Macro F1-Score : {f1:.4f}")
print("=" * 50)
print(classification_report(labels, preds, target_names=GESTURE_CLASSES))

# Confusion Matrix
cm = confusion_matrix(labels, preds)
short_names = [c.replace(" ", "\n") for c in GESTURE_CLASSES]

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=short_names, yticklabels=short_names)
plt.title("Confusion Matrix — CNN-LSTM", fontsize=14)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("/kaggle/working/confusion_matrix.png", dpi=150)
plt.show()

# Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs = range(1, NUM_EPOCHS + 1)
ax1.plot(epochs, history["train_acc"], label="Train", marker="o")
ax1.plot(epochs, history["val_acc"],   label="Val",   marker="o")
ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch")
ax1.legend(); ax1.grid(True)

ax2.plot(epochs, history["train_loss"], label="Train", marker="o")
ax2.plot(epochs, history["val_loss"],   label="Val",   marker="o")
ax2.set_title("Loss"); ax2.set_xlabel("Epoch")
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig("/kaggle/working/training_curves.png", dpi=150)
plt.show()

# ── HÜCRE 7: Model Analizi ───────────────────────────────────────
# Model boyutu
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
size_mb = os.path.getsize("/kaggle/working/cnn_lstm_best.pth") / (1024**2)

print(f"Model Boyutu   : {size_mb:.2f} MB")
print(f"Total Params   : {total_params:,}")
print(f"Trainable      : {trainable_params:,}")

# Inference latency (CPU'da ölç — gerçek kullanım senaryosu)
model_cpu = model.cpu().eval()
sample = next(iter(val_loader))[0][:1]  # 1 sample, CPU

# Warm-up
with torch.no_grad():
    for _ in range(3):
        model_cpu(sample)

times = []
with torch.no_grad():
    for _ in range(20):
        t0 = time.perf_counter()
        model_cpu(sample)
        times.append((time.perf_counter() - t0) * 1000)

latency_mean = np.mean(times)
latency_std  = np.std(times)
print(f"Inference (CPU): {latency_mean:.1f} ± {latency_std:.1f} ms")

print("\n=== Özet ===")
print(f"Top-1 Accuracy : {test_acc:.4f}")
print(f"Macro F1       : {f1:.4f}")
print(f"Model Size     : {size_mb:.2f} MB")
print(f"Latency (CPU)  : {latency_mean:.1f} ms")
