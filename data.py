import os
import numpy as np
import h5py
import cv2

INPUT_DIR = "./data_raw"
OUTPUT_DIR = "./data"

label_map = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary"
}

# --- 1. Initialize Tracking Variables ---
stats = {
    "total_images": 0,
    "class_counts": {1: 0, 2: 0, 3: 0},
    "patient_counts": {}  # Will map PID -> number of slices
}

# Create folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
for cls in label_map.values():
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

def normalize_image(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)

print("Starting extraction...")

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".mat"):
        continue

    path = os.path.join(INPUT_DIR, file)

    try:
        with h5py.File(path, 'r') as f:
            cjdata = f['cjdata']

            # Extract fields
            image = np.array(cjdata['image']).T
            label = int(np.array(cjdata['label'])[0][0])
            
            # Corrected PID extraction
            pid_array = np.array(cjdata['PID']).flatten()
            pid = "".join([chr(c) for c in pid_array])

            # --- 2. Update Stats ---
            stats["total_images"] += 1
            stats["class_counts"][label] += 1
            stats["patient_counts"][pid] = stats["patient_counts"].get(pid, 0) + 1

            # Normalize image
            image = normalize_image(image)
            class_name = label_map[label]

            # Save image
            filename = f"pid_{pid}_{file.replace('.mat', '.jpg')}"
            save_path = os.path.join(OUTPUT_DIR, class_name, filename)

            cv2.imwrite(save_path, image)

    except Exception as e:
        print(f"Skipping {file}: {e}")

# --- 3. Print Sanity Checks ---
print("\n" + "="*40)
print("       EXTRACTION SANITY CHECKS")
print("="*40)

print(f"Total Images Extracted: {stats['total_images']}")
print(f"Total Unique Patients:  {len(stats['patient_counts'])}")

print("\n--- Images per Class ---")
for label_idx, count in stats['class_counts'].items():
    print(f"[{label_map[label_idx].capitalize()}]: {count} slices")

print("\n--- Top 5 Patients by Slice Count ---")
# Sort the patient dictionary by the number of slices (descending)
sorted_patients = sorted(stats['patient_counts'].items(), key=lambda item: item[1], reverse=True)

for i, (pid, count) in enumerate(sorted_patients[:5], start=1):
    print(f"{i}. PID {pid}: {count} slices")

print("="*40)