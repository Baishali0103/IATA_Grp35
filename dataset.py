import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import size2id, color2id, shape2id, relation2id, get_image_path


class ShapesDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = get_image_path(self.image_dir, row["sample_id"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        labels = {
            "obj1_size": torch.tensor(size2id[row["obj1_size"]], dtype=torch.long),
            "obj1_color": torch.tensor(color2id[row["obj1_color"]], dtype=torch.long),
            "obj1_shape": torch.tensor(shape2id[row["obj1_shape"]], dtype=torch.long),
            "relation": torch.tensor(relation2id[row["relation"]], dtype=torch.long),
            "obj2_size": torch.tensor(size2id[row["obj2_size"]], dtype=torch.long),
            "obj2_color": torch.tensor(color2id[row["obj2_color"]], dtype=torch.long),
            "obj2_shape": torch.tensor(shape2id[row["obj2_shape"]], dtype=torch.long),
        }

        sentence = row["sentence"]

        return image, labels, sentence