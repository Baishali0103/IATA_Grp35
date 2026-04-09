import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class TicTacToeDataset(Dataset):
    def __init__(self, label_file, img_dir):
        self.data = []
        self.transform = transforms.ToTensor()
        self.img_dir = img_dir

        with open(label_file) as f:
            for line in f:
                img, label, sentence = line.strip().split("\t")
                label = list(map(int, label.split()))
                self.data.append((img, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        image = Image.open(f"{self.img_dir}/{img_name}")
        image = self.transform(image)

        label = torch.tensor(label)
        return image, label