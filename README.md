# ChipGAN-PyTorch1.8-
**Original Code & Paper**
This is the latest code for paper "ChipGAN: A Generative Adversarial Network for Chinese Ink Wash Painting Style Transfer". Original Work by Bin He, Feng Gao etc.
You can check the original code and paper at: [CODE](https://github.com/PKU-IMRE/ChipGAN) | [PAPER](https://dl.acm.org/doi/10.1145/3240508.3240655)

## Description
In the paper above, He provides a new way to generate Chinese ink wash painting by using Generative Adversarial Network(GAN). You only have to input a real-world image, then the pre-trained GAN will generate an ink wash painting for you.

## Requirement Library 
PyTorch 1.8 | OS: Windows 10 | Python 3.7

## Training Datasets
In this code, two datasets are needed to train your GAN. One is called the landscape dataset(DatasetA), another is called the ink wash paintings dataset(DatasetB).

For the DatasetA, I choose them from [Kaggle|Landscape Pictures](https://www.kaggle.com/arnaud58/landscape-pictures).

For the DatasetB, I choose them from [Traditional Chinese Landscape Painting Dataset](https://github.com/alicex2020/Chinese-Landscape-Painting-Dataset)

You could train your GAN by your own or use the pre-trained GAN in the folder "saved_models".
