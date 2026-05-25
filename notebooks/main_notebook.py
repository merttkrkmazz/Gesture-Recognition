# ============================================================
# CS466 — Gesture Recognition: Comparative Analysis
# Bu dosyayı Kaggle notebook'a hücre hücre kopyala.
# ============================================================

# ── HÜCRE 1: Kurulum ──────────────────────────────────────
# !pip install mediapipe -q

# ── HÜCRE 2: Import + Config ──────────────────────────────
import os, sys
sys.path.append("/kaggle/working")

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

GESTURE_CLASSES = [
    "Swiping Left", "Swiping Right", "Swiping Up", "Swiping Down",
    "Thumb Up", "Thumb Down",
    "Zooming In With Full Hand", "Zooming Out With Full Hand",
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(GESTURE_CLASSES)}
NUM_CLASSES = len(GESTURE_CLASSES)
NUM_FRAMES  = 37
BATCH_SIZE  = 16
NUM_EPOCHS  = 20
LR          = 1e-4

# Kaggle'daki dataset yolunu kontrol et
# !ls /kaggle/input/
DATA_ROOT  = "/kaggle/input/20bn-jester-dataset-v1/20BN-JESTER-V1"
TRAIN_CSV  = "/kaggle/input/20bn-jester-dataset-v1/jester-v1-train.csv"
VAL_CSV    = "/kaggle/input/20bn-jester-dataset-v1/jester-v1-validation.csv"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ── HÜCRE 3: Dataset yolunu doğrula ───────────────────────
# !ls /kaggle/input/20bn-jester-dataset-v1/ | head -20

# ── HÜCRE 4: DataLoader'ları oluştur ──────────────────────
from data.dataset import FrameSequenceDataset, VideoDataset

# CNN-LSTM için
train_seq = FrameSequenceDataset(TRAIN_CSV, DATA_ROOT, CLASS_TO_IDX,
                                  num_frames=NUM_FRAMES, img_size=224)
val_seq   = FrameSequenceDataset(VAL_CSV,   DATA_ROOT, CLASS_TO_IDX,
                                  num_frames=NUM_FRAMES, img_size=224)

train_loader_seq = DataLoader(train_seq, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=2, pin_memory=True)
val_loader_seq   = DataLoader(val_seq,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=2, pin_memory=True)

# 3D CNN için
train_vid = VideoDataset(TRAIN_CSV, DATA_ROOT, CLASS_TO_IDX,
                          num_frames=NUM_FRAMES, img_size=112)
val_vid   = VideoDataset(VAL_CSV,   DATA_ROOT, CLASS_TO_IDX,
                          num_frames=NUM_FRAMES, img_size=112)

train_loader_vid = DataLoader(train_vid, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=2, pin_memory=True)
val_loader_vid   = DataLoader(val_vid,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=2, pin_memory=True)

print(f"Train samples: {len(train_seq)}, Val samples: {len(val_seq)}")

# ── HÜCRE 5: Landmark extraction (bir kere çalıştır) ──────
from data.extract_landmarks import extract_landmarks_for_split

LANDMARK_DIR = "/kaggle/working/landmarks"
extract_landmarks_for_split(TRAIN_CSV, DATA_ROOT, LANDMARK_DIR, CLASS_TO_IDX)
extract_landmarks_for_split(VAL_CSV,   DATA_ROOT, LANDMARK_DIR, CLASS_TO_IDX)

# ── HÜCRE 6: Landmark DataLoader ──────────────────────────
from data.dataset import LandmarkDataset

train_lm = LandmarkDataset(TRAIN_CSV, LANDMARK_DIR, CLASS_TO_IDX)
val_lm   = LandmarkDataset(VAL_CSV,   LANDMARK_DIR, CLASS_TO_IDX)

train_loader_lm = DataLoader(train_lm, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2)
val_loader_lm   = DataLoader(val_lm,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)

# ── HÜCRE 7: Model 1 — Landmark BiLSTM ────────────────────
from models.landmark_rnn import LandmarkBiLSTM
from train import train_model

model_lm = LandmarkBiLSTM(num_classes=NUM_CLASSES)
history_lm = train_model(model_lm, train_loader_lm, val_loader_lm,
                          DEVICE, num_epochs=NUM_EPOCHS,
                          model_name="landmark_bilstm")

# ── HÜCRE 8: Model 2 — CNN-LSTM ───────────────────────────
from models.cnn_lstm import CNNLSTM

model_cnn = CNNLSTM(num_classes=NUM_CLASSES)
history_cnn = train_model(model_cnn, train_loader_seq, val_loader_seq,
                           DEVICE, num_epochs=NUM_EPOCHS,
                           model_name="cnn_lstm")

# ── HÜCRE 9: Model 3 — R3D-18 ─────────────────────────────
from models.r3d_model import R3DGesture

model_r3d = R3DGesture(num_classes=NUM_CLASSES)
history_r3d = train_model(model_r3d, train_loader_vid, val_loader_vid,
                           DEVICE, num_epochs=NUM_EPOCHS,
                           model_name="r3d18")

# ── HÜCRE 10: Evaluation & Karşılaştırma ──────────────────
from evaluate import full_evaluation, plot_comparison, plot_training_history

criterion = nn.CrossEntropyLoss()

# Her model için best checkpoint'i yükle
model_lm.load_state_dict(torch.load("landmark_bilstm_best.pth"))
model_cnn.load_state_dict(torch.load("cnn_lstm_best.pth"))
model_r3d.load_state_dict(torch.load("r3d18_best.pth"))

# Sample input'lar latency ölçümü için (batch_size=1)
sample_lm  = next(iter(val_loader_lm))[0][:1]
sample_cnn = next(iter(val_loader_seq))[0][:1]
sample_r3d = next(iter(val_loader_vid))[0][:1]

results = [
    full_evaluation(model_lm,  val_loader_lm,  criterion, DEVICE,
                    "Landmark-BiLSTM", sample_lm),
    full_evaluation(model_cnn, val_loader_seq, criterion, DEVICE,
                    "CNN-LSTM",         sample_cnn),
    full_evaluation(model_r3d, val_loader_vid, criterion, DEVICE,
                    "R3D-18",           sample_r3d),
]

plot_comparison(results)
plot_training_history([history_lm, history_cnn, history_r3d],
                      ["Landmark-BiLSTM", "CNN-LSTM", "R3D-18"])
