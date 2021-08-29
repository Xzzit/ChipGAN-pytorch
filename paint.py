import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os
import random
import config
from generator import Generator

# Create Generator B which paints ink wash painting.
gen_B = Generator(img_channels=3).to(config.DEVICE)
checkpoint = torch.load('E:/project/Python/ChipGAN/saved_models/genB.pth.tar', map_location=config.DEVICE)
gen_B.load_state_dict(checkpoint["state_dict"])

# Create Data loader.
root_A = 'E:/project/Python/dataset/inkpainting/trainA'     # root_A is the directory you save your real world images.

my_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomGrayscale(p=0),
    transforms.ToTensor(),
])
selected_A = random.choices(os.listdir(root_A), k=10)   # Randomly select k=10 images
filelist = []
for i in selected_A:
    filelist.append(os.path.join(root_A, i))
A_img = []
for i in filelist:
    img = my_transform(Image.open(i))
    save_image(img, f'painting/{i[44:]}.png')   # Start at location 44 to ignore root_A and just save files name.
    img.unsqueeze_(0)   # Add one more dimension to tensor. [3, 256, 256] extended to [1, 3, 256, 256] in this case.
    A_img.append(img)
A_img = torch.cat(A_img, dim=0)  # Combine all k=10 tensors together.

# Create Painting
A_img = A_img.to('cuda')
gen_B = gen_B.to('cuda')
fake_img = gen_B(A_img)

for idx in range(fake_img.shape[0]):
    save_image(fake_img[idx], f'painting/B_{idx}.png')  # Save all generated images.
