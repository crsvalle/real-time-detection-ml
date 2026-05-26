from pathlib import Path
from scipy.io import loadmat


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data/raw/stanford-cars/car_devkit/devkit"

ann_path = DATA_PATH / "cars_train_annos.mat"

print("Using path:", ann_path)

annos = loadmat(ann_path)
annotations = annos["annotations"][0]
data = []

for ann in annotations:
    class_id = int(ann[4][0][0]) - 1  # fix to 0–195
    img_name = ann[5][0]

    data.append((img_name, class_id))

print("Processed samples:", len(data))
print("Example:", data[0])