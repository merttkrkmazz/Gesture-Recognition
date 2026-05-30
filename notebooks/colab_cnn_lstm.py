# ================================================================
# CS466 — Gesture Recognition: CNN-LSTM Model (Google Colab)
# ================================================================

# ── HÜCRE 1: Drive bağla ve zip'i çıkar ─────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import zipfile, os

ZIP_PATH    = "/content/drive/MyDrive/Jester20bn_subset.zip"  # Drive'daki zip yolu
EXTRACT_DIR = "/content/"

if not os.path.exists("/content/Jester20bn_subset"):
    print("Zip çıkarılıyor...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(EXTRACT_DIR)
    print("Tamamlandı.")
else:
    print("Dataset zaten mevcut.")

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
IMG_SIZE     = 112
BATCH_SIZE   = 16
NUM_EPOCHS   = 15
LR           = 1e-4
WEIGHT_DECAY = 1e-4
SEED         = 42

BASE       = "/content/Jester20bn_subset"
TRAIN_ROOT = f"{BASE}/train_subset"
VAL_ROOT   = f"{BASE}/val_subset"
TRAIN_CSV  = f"{BASE}/train_subset/train_subset.csv"
VAL_CSV    = f"{BASE}/val_subset/val_subset.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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
    return [(str(int(r["video_id"])), CLASS_TO_IDX[r["label"].strip()])
            for _, r in df.iterrows() if r["label"].strip() in CLASS_TO_IDX]


def load_frames(video_dir):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    files = sorted(
        [f for f in os.listdir(video_dir) if f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    frames = [transform(Image.open(os.path.join(video_dir, f)).convert("RGB"))
              for f in files[:NUM_FRAMES]]
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])
    return torch.stack(frames)


class GestureDataset(Dataset):
    def __init__(self, csv_path, data_root):
        self.samples = load_split(csv_path)
        self.root    = data_root
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        vid, label = self.samples[idx]
        return load_frames(os.path.join(self.root, vid)), label


train_ds = GestureDataset(TRAIN_CSV, TRAIN_ROOT)
val_ds   = GestureDataset(VAL_CSV,   VAL_ROOT)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── HÜCRE 4: Model ───────────────────────────────────────────────
import torch.nn as nn
from torchvision import models


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.feat_dim = bb.last_channel
        bb.classifier = nn.Identity()
        self.cnn = bb
        for p in self.cnn.features[:-3].parameters():
            p.requires_grad = False
        self.lstm = nn.LSTM(self.feat_dim, 256, 2,
                            batch_first=True, dropout=0.3)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))

    def forward(self, x):
        B, T, C, H, W = x.shape
        f = self.cnn(x.view(B*T, C, H, W)).view(B, T, -1)
        out, _ = self.lstm(f)
        return self.head(out[:, -1])


model = CNNLSTM().to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

# ── HÜCRE 5: Train ───────────────────────────────────────────────
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_epoch(model, loader, opt, crit):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.item() * y.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum/total, correct/total


@torch.no_grad()
def eval_epoch(model, loader, crit):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss_sum += crit(out, y).item() * y.size(0)
        p = out.argmax(1)
        correct += (p == y).sum().item()
        total   += y.size(0)
        preds.extend(p.cpu().tolist())
        trues.extend(y.cpu().tolist())
    return loss_sum/total, correct/total, preds, trues


crit  = nn.CrossEntropyLoss()
opt   = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
              lr=LR, weight_decay=WEIGHT_DECAY)
sched = CosineAnnealingLR(opt, T_max=NUM_EPOCHS)

history  = {"tr_loss":[], "tr_acc":[], "vl_loss":[], "vl_acc":[]}
best_acc = 0

SAVE_PATH = "/content/drive/MyDrive/cnn_lstm_best.pth"  # Drive'a kaydet

print(f"{'Ep':>3} {'TrLoss':>8} {'TrAcc':>7} {'VlLoss':>8} {'VlAcc':>7} {'Time':>6}")
for ep in range(1, NUM_EPOCHS+1):
    t0 = time.time()
    tl, ta = train_epoch(model, train_loader, opt, crit)
    vl, va, _, _ = eval_epoch(model, val_loader, crit)
    sched.step()
    for k,v in zip(["tr_loss","tr_acc","vl_loss","vl_acc"],[tl,ta,vl,va]):
        history[k].append(v)
    flag = " *" if va > best_acc else ""
    if va > best_acc:
        best_acc = va
        torch.save(model.state_dict(), SAVE_PATH)
    print(f"{ep:>3} {tl:>8.4f} {ta:>7.4f} {vl:>8.4f} {va:>7.4f} "
          f"{time.time()-t0:>5.0f}s{flag}")

print(f"\nBest Val Acc: {best_acc:.4f}")
print(f"Model kaydedildi: {SAVE_PATH}")

# ── HÜCRE 6: Evaluation ──────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

model.load_state_dict(torch.load(SAVE_PATH))
_, acc, preds, trues = eval_epoch(model, val_loader, crit)
f1 = f1_score(trues, preds, average="macro")

print(f"Top-1 Accuracy : {acc:.4f}")
print(f"Macro F1-Score : {f1:.4f}")
print(classification_report(trues, preds, target_names=GESTURE_CLASSES))

cm = confusion_matrix(trues, preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[c.replace(" ","\n") for c in GESTURE_CLASSES],
            yticklabels=[c.replace(" ","\n") for c in GESTURE_CLASSES])
plt.title("Confusion Matrix — CNN-LSTM")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/confusion_matrix.png", dpi=150)
plt.show()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.plot(history["tr_acc"], label="Train"); a1.plot(history["vl_acc"], label="Val")
a1.set_title("Accuracy"); a1.legend(); a1.grid()
a2.plot(history["tr_loss"], label="Train"); a2.plot(history["vl_loss"], label="Val")
a2.set_title("Loss"); a2.legend(); a2.grid()
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/training_curves.png", dpi=150)
plt.show()

# ── HÜCRE 7: Model Analizi ───────────────────────────────────────
size_mb  = os.path.getsize(SAVE_PATH) / (1024**2)
total_p  = sum(p.numel() for p in model.parameters())

model_cpu = model.cpu().eval()
sample = next(iter(val_loader))[0][:1]
with torch.no_grad():
    for _ in range(3): model_cpu(sample)
times = []
with torch.no_grad():
    for _ in range(20):
        t0 = time.perf_counter()
        model_cpu(sample)
        times.append((time.perf_counter()-t0)*1000)

print(f"Model Size  : {size_mb:.2f} MB")
print(f"Total Params: {total_p:,}")
print(f"Latency(CPU): {np.mean(times):.1f} ± {np.std(times):.1f} ms")
print(f"\n=== Özet ===")
print(f"Top-1 Accuracy : {acc:.4f}")
print(f"Macro F1       : {f1:.4f}")
print(f"Model Size     : {size_mb:.2f} MB")
print(f"Latency (CPU)  : {np.mean(times):.1f} ms")
