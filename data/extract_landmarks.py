"""
MediaPipe ile landmark'ları önceden çıkarır ve .pt olarak kaydeder.
Kaggle notebook'ta bir kere çalıştır, sonra LandmarkDataset kullan.
"""
import os
import csv
import torch
import numpy as np
from PIL import Image
import mediapipe as mp


def extract_landmarks_for_split(csv_path, data_root, output_dir,
                                class_to_idx, num_frames=37):
    os.makedirs(output_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    )

    with open(csv_path, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        rows = [(r[0].strip(), r[1].strip()) for r in reader
                if len(r) >= 2 and r[1].strip() in class_to_idx]

    print(f"{len(rows)} video işlenecek...")

    for i, (video_id, _) in enumerate(rows):
        out_path = os.path.join(output_dir, f"{video_id}.pt")
        if os.path.exists(out_path):
            continue

        video_dir = os.path.join(data_root, video_id)
        frame_files = sorted(
            [f for f in os.listdir(video_dir) if f.endswith((".jpg", ".png"))],
            key=lambda x: int(os.path.splitext(x)[0])
        )

        sequence = []
        last_valid = np.zeros(63, dtype=np.float32)

        for fname in frame_files[:num_frames]:
            img = np.array(Image.open(os.path.join(video_dir, fname)).convert("RGB"))
            result = hands.process(img)

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark
                coords = np.array([[p.x, p.y, p.z] for p in lm],
                                  dtype=np.float32).flatten()
                last_valid = coords
            else:
                coords = last_valid.copy()

            sequence.append(coords)

        # Pad
        while len(sequence) < num_frames:
            sequence.append(last_valid.copy())

        tensor = torch.tensor(np.stack(sequence))  # (T, 63)
        torch.save(tensor, out_path)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(rows)} tamamlandı")

    hands.close()
    print("Landmark extraction bitti.")
