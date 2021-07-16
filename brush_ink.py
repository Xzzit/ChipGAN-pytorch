import torch
import torch.nn as nn
import config
import math
import torch.nn.functional as F


class HED(nn.Module):
	def __init__(self):
		super(HED, self).__init__()

		self.netVggOne = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.netVggTwo = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.netVggThr = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.netVggFou = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.netVggFiv = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.netScoreOne = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreTwo = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreThr = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreFou = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreFiv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

		self.netCombine = nn.Sequential(
			nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid()
		)

		self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(config.HED_MODEL_DIR).items()})
	# end

	def forward(self, tenInput):
		tenBlue = (tenInput[:, 0:1, :, :] * 255.0) - 104.00698793
		tenGreen = (tenInput[:, 1:2, :, :] * 255.0) - 116.66876762
		tenRed = (tenInput[:, 2:3, :, :] * 255.0) - 122.67891434

		tenInput = torch.cat([ tenBlue, tenGreen, tenRed ], 1)

		tenVggOne = self.netVggOne(tenInput)
		tenVggTwo = self.netVggTwo(tenVggOne)
		tenVggThr = self.netVggThr(tenVggTwo)
		tenVggFou = self.netVggFou(tenVggThr)
		tenVggFiv = self.netVggFiv(tenVggFou)

		tenScoreOne = self.netScoreOne(tenVggOne)
		tenScoreTwo = self.netScoreTwo(tenVggTwo)
		tenScoreThr = self.netScoreThr(tenVggThr)
		tenScoreFou = self.netScoreFou(tenVggFou)
		tenScoreFiv = self.netScoreFiv(tenVggFiv)

		tenScoreOne = nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreTwo = nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreThr = nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreFou = nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreFiv = nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

		return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))


def no_sigmoid_cross_entropy(sig_logits, label):
    count_neg = torch.sum(1.-label)
    count_pos = torch.sum(label)

    beta = count_neg / (count_pos+count_neg)
    pos_weight = beta / (1-beta)

    cost = pos_weight * label * (-1) * torch.log(sig_logits) + (1-label)* (-1) * torch.log(1-sig_logits)
    cost = torch.mean(cost * (1-beta))

    return cost


def gauss_kernel(image, kernel_size=21, sigma=3, channels=3):
	# Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
	x_cord = torch.arange(kernel_size)
	x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
	y_grid = x_grid.t()
	xy_grid = torch.stack([x_grid, y_grid], dim=-1)

	mean = (kernel_size - 1)/2.
	variance = sigma**2.

	# Calculate the 2-dimensional gaussian kernel which is
	# the product of two gaussian distributions for two different
	# variables (in this case called x and y)
	gaussian_kernel = (1./(2.*math.pi*variance)) *\
					torch.exp(
						-torch.sum((xy_grid - mean)**2., dim=-1) /\
						(2*variance)
					).cuda()

	# Make sure sum of values in gaussian kernel equals 1.
	gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

	# Reshape to 2d depthwise convolutional weight
	gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
	gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

	gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
								kernel_size=kernel_size, groups=channels, bias=False)
	
	gaussian_filter.weight.data = gaussian_kernel
	gaussian_filter.weight.requires_grad = False

	return gaussian_filter(image).cuda()


def erode(image, kernel_size=5):
	pad_size = int(kernel_size/2)
	p1d = (pad_size,pad_size,pad_size,pad_size)
	padding_image = F.pad(image, p1d, "constant", 1)
	erode_image = -1 * (F.max_pool2d(-1 * padding_image, kernel_size, 1))

	return erode_image


def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((1, img_channels, img_size+200, img_size))
    hed = HED()
    print(hed(x).shape)


if __name__ == "__main__":
    test()
