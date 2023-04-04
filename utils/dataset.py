from PIL import Image
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class ABDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

        self.A_images = os.listdir(root_A)
        self.B_images = os.listdir(root_B)
        self.length_dataset = max(len(self.B_images), len(self.A_images))
        self.A_len = len(self.A_images)
        self.B_len = len(self.B_images)
        
    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        B_img = self.B_images[index % self.B_len]
        A_img = self.A_images[index % self.A_len]

        B_path = os.path.join(self.root_B, B_img)
        A_path = os.path.join(self.root_A, A_img)

        B_img = np.array(Image.open(B_path).convert("RGB"))
        assert not np.any(np.isnan(B_img))
        A_img = np.array(Image.open(A_path).convert("RGB"))
        assert not np.any(np.isnan(A_img))

        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)

        return [A_img, B_img]


my_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])