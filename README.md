# ChipGAN-PyTorch
**Visit the Original Code & Paper**:
This is the latest code for paper "ChipGAN: A Generative Adversarial Network for Chinese Ink Wash Painting Style Transfer". Original work by Bin He, Feng Gao etc.

Check the original code and paper at: [CODE](https://github.com/PKU-IMRE/ChipGAN) | [PAPER](https://dl.acm.org/doi/10.1145/3240508.3240655)

## Description
In the paper above, He provides a new way to generate Chinese ink wash painting by using Generative Adversarial Network(GAN). The core modules of ChipGAN enforce three constraints – voids, brush strokes, and ink wash tone and diffusion – to address three key techniques commonly adopted in Chinese ink wash painting.

## Dependencies
Tested with:
* OS: Windows 10/11 or Ubuntu 22.04 (Recommended)
* PyTorch 1.13 
* Python 3.7

## Create Environment
```
git clone https://github.com/Xzzit/ChipGAN-pytorch.git
cd ChipGAN-pytorch
conda create -n chipgan python=3.7
conda activate chipgan
pip install -r requirements.txt
```
## Painting with Pre-trained Models
Download pre-trained models from [ChipGAN Models](https://drive.google.com/drive/folders/1lzS3LVWfSYo8viaLLJpoKeQHSrMqMwt5?usp=sharing)

The downloaded file should be placed as following:
```
.
├── ...
├── saved_models                # Store all pre-trained models
│   ├── criticA.pth.tar             # Detect whether a image is a Landscape paintings or not
│   ├── criticB.pth.tar             # Detect whether a image is a Inkwash paintings or not
│   ├── criticINK.pth.tar           # Detect whether a image is a Blurred Inkwash painting or not
│   ├── genA.pth.tar                # Take inkwash painting as input and generate Landscape paintings
│   ├── genB.pth.tar                # Take Landscape painting as input and generate inkwash paintings
│   ├── hed-bsds500                 # Edge detection & blur
│   └── ...                         # Ohter models trained by user are also stored in here
└── saved_images                # Store all input and generated images
```

Run `paint.py` file to generate your ink wash painting images.

In this file, `img_dir` should be changed to the input image directory.

Open the file to read more details and I have written code comments for you. Have fun!


## Training ChipGAN Models
Two datasets are needed to train ChipGAN. One is the landscape dataset(DatasetA), another is the ink wash paintings dataset(DatasetB).

For the DatasetA, I choose them from [Kaggle|Landscape Pictures](https://www.kaggle.com/arnaud58/landscape-pictures).

For the DatasetB, I choose them from [Traditional Chinese Landscape Painting Dataset](https://github.com/alicex2020/Chinese-Landscape-Painting-Dataset)

The downloaded file should be placed as following:

```
.
├── ...
├── saved_models
    ├── ...
├── saved_images
    ├── ...
└── data                        # Store all training images
    ├── trainA                      # Dataset of landscape
    └── trainB                      # Dataset of ink wash painting
```

Run `train.py` to train ChipGAN models.