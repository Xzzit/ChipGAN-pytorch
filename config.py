import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "E:/project/Python/dataset/inkpainting"
HED_MODEL_DIR = 'E:/project/Python/models/hed-bsds500'
CHECKPOINT_GEN_A = "saved_models/genA.pth.tar"
CHECKPOINT_GEN_B = "saved_models/genB.pth.tar"
CHECKPOINT_CRITIC_A = "saved_models/criticA.pth.tar"
CHECKPOINT_CRITIC_B = "saved_models/criticB.pth.tar"
CHECKPOINT_CRITIC_INK = "saved_models/criticINK.pth.tar"

BATCH_SIZE = 1
LEARNING_RATE = 1e-6
LAMBDA_IDENTITY = 0
LAMBDA_CYCLE = 10
LAMBDA_BRUSH = 10
LAMBDA_INK = 0.05
NUM_WORKERS = 0
NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = True
