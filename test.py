import torch
import torch.nn as nn
import config
from brush_ink import no_sigmoid_cross_entropy
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from torchvision.utils import save_image
from brush_ink import gauss_kernel, erode
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision.transforms as transforms
from dataset import ABDataset, my_transform


# Test for HED downloader
# a = {strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(config.HED_MODEL_DIR).items()}
# for i in a:
#     print(i)





# Test for no_sigmoid_cross_entropy function
# x = torch.randn((1, 3, 244, 244))
# y = torch.randn((1, 3, 244, 244))
# edge_real_A = torch.sigmoid(x)
# edge_fake_B = torch.sigmoid(y)
# print(edge_fake_B.shape)
# loss_edge_1 = no_sigmoid_cross_entropy(edge_fake_B, edge_real_A)
# print(loss_edge_1)





# Test for Gaussian kernel
# image = Image.open('horse.jpg')
# tenInput = F.to_tensor(image)
# tenInput = tenInput.unsqueeze(0)
# print(tenInput.shape)
# tenOutput = gauss_kernel(tenInput)
# print(tenOutput.shape)
# save_image(tenOutput, '1.jpg')





# Test for erode
# image = Image.open('horse.jpg')
# tenInput = TF.to_tensor(image)
# tenInput = tenInput.unsqueeze(0)

# tenOutput = erode(tenInput)
# tenOutput = gauss_kernel(tenOutput)
# print(tenOutput.shape)

# save_image(tenOutput, '1.jpg')





# Test for transformer
# transforms = A.Compose(
#     [
#         A.Resize(width=256, height=256),
#         A.Normalize(),
#         ToTensorV2(),
#     ],
#     additional_targets={"image0": "image"},
# )

# B_img = np.array(Image.open('a.jpg').convert("RGB"))
# B_img = transforms(image=B_img)
# print(B_img['image'].shape)
# save_image(B_img['image'], '1.png')

# img = Image.open('a.jpg')
# img = ToTensor()(img)
# out = F.interpolate(img, size=256)  #The resize operation on tensor.
# ToPILImage()(out).save('1.png', mode='png')


# my_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 256)),
#     # transforms.RandomCrop((224, 224)),
#     # transforms.ColorJitter(brightness=0.5),
#     # transforms.RandomRotation(degrees=45),
#     # transforms.RandomHorizontalFlip(p=0.5),
#     # transforms.RandomVerticalFlip(p=0.05),
#     # transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor(),
# ])

# B_img = np.array(Image.open('a.jpg').convert("RGB"))
# B_img = my_transform(B_img)
# print(B_img.shape)
# save_image(B_img, '1.png')



# Test for Dataloader
dataset = ABDataset(
    root_A=config.TRAIN_DIR + "/trainA", root_B=config.TRAIN_DIR + "/trainB", transform=my_transform
)

loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)

for a,b in loader:
    print(a.shape)
    save_image(b, '1.jpg')
    break