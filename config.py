GESTURE_CLASSES = [
    "Swiping Left",
    "Swiping Right",
    "Swiping Up",
    "Swiping Down",
    "Thumb Up",
    "Thumb Down",
    "Zooming In With Full Hand",
    "Zooming Out With Full Hand",
]

CLASS_TO_IDX = {cls: i for i, cls in enumerate(GESTURE_CLASSES)}
NUM_CLASSES = len(GESTURE_CLASSES)

# Input shape
NUM_FRAMES = 37
IMG_SIZE = 112          # R3D-18 için 112x112, CNN-LSTM için 224x224
LANDMARK_DIM = 63       # 21 keypoints x 3 (x, y, z)

# Training
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Kaggle paths
DATA_ROOT_TRAIN = "/kaggle/input/datasets/toxicmender/20bn-jester/Train"
DATA_ROOT_VAL   = "/kaggle/input/datasets/toxicmender/20bn-jester/Validation"
TRAIN_CSV       = "/kaggle/input/datasets/toxicmender/20bn-jester/Train.csv"
VAL_CSV         = "/kaggle/input/datasets/toxicmender/20bn-jester/Validation.csv"
