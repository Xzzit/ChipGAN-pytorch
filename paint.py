import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import config
from utils.generator import Generator

# Create Generator B which paints ink wash painting.
gen_B = Generator(img_channels=3).to(config.DEVICE)
checkpoint = torch.load('saved_models/genB.pth.tar', map_location=config.DEVICE)
gen_B.load_state_dict(checkpoint["state_dict"])

# Image directory
img_dir = 'saved_images/mountain.jpg'

# Create img transformer.
my_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomGrayscale(p=0),
    transforms.ToTensor(),
])

# Load image
img = my_transform(Image.open(img_dir))
img.unsqueeze_(0)   # Add one more dimension to tensor. [3, 256, 256] extended to [1, 3, 256, 256] in this case.

# Create Painting
img = img.to('cuda')
gen_B = gen_B.to('cuda')
fake_img = gen_B(img)

# Save image
save_image(fake_img, f'saved_images/inkwash.png')
