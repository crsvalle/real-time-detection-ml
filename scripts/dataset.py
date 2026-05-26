from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class CarDataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = Path(img_dir).resolve()  # 🔥 FORCE ABSOLUTE PATH
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]

        img_path = self.img_dir / img_name

        # 🔥 DEBUG (temporarily)
        # print(img_path)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
