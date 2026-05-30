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

NUM_FRAMES  = 37
IMG_SIZE    = 224

BATCH_SIZE  = 16
NUM_EPOCHS  = 15
LR          = 1e-4
WEIGHT_DECAY = 1e-4

# Kaggle paths (zip Kaggle'a yüklendikten sonra güncelle)
BASE        = "/kaggle/input/jester-subset/Jester20bn_subset"
TRAIN_ROOT  = f"{BASE}/train_subset"
VAL_ROOT    = f"{BASE}/val_subset"
TRAIN_CSV   = f"{BASE}/train_subset/train_subset.csv"
VAL_CSV     = f"{BASE}/val_subset/val_subset.csv"
