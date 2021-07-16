import torch
from dataset import ABDataset, my_transform
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
from brush_ink import HED, no_sigmoid_cross_entropy, gauss_kernel, erode


'''
In this case, A stands for real world image and B stands for ink wash painting.
disc_A: If it is A or not.
gen_A: To generate fake A.
'''
def train(disc_A, disc_B, disc_ink, gen_A, gen_B, hed, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    B_reals = 0
    B_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (A, B) in enumerate(loop):
        A = A.to(config.DEVICE)
        B = B.to(config.DEVICE)

        fake_A = gen_A(B)
        fake_B = gen_B(A)

        if idx % 200 == 0:
            save_image(A, f'saved_images/A_{idx}.png')
            save_image(B, f'saved_images/B_{idx}.png')
            save_image(fake_A, f'saved_images/Fake_A_{idx}.png')
            save_image(fake_B, f'saved_images/Fake_B_{idx}.png')


        # TODO: Train Discriminator A, B and ink

        # Train Discriminator A
        D_A_real = disc_A(A)
        D_A_fake = disc_A(fake_A.detach())  # Here we don't want to touch generator
        D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
        D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
        D_A_loss = D_A_real_loss + D_A_fake_loss

        # Train Discriminator B
        D_B_real = disc_B(B)
        D_B_fake = disc_B(fake_B.detach())
        B_reals += D_B_real.mean().item()
        B_fakes += D_B_fake.mean().item()
        D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
        D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
        D_B_loss = D_B_real_loss + D_B_fake_loss

        # Train Discriminator ink
        ink_B = gauss_kernel(erode(B))
        ink_fake_B = gauss_kernel(erode(fake_B.detach()))
        D_ink_real = disc_ink(ink_B)
        D_ink_fake = disc_ink(ink_fake_B)
        D_ink_real_loss = mse(D_ink_real, torch.ones_like(D_ink_real))
        D_ink_fake_loss = mse(D_ink_fake, torch.zeros_like(D_ink_fake))
        D_ink_loss = D_ink_real_loss + D_ink_fake_loss

        # put it together
        D_loss = D_A_loss + D_B_loss + D_ink_loss * config.LAMBDA_INK

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()


        # TODO: Train Generator A and B

        # adversarial loss for both generators
        D_A_fake = disc_A(fake_A)
        D_B_fake = disc_B(fake_B)
        ink_fake_B = gauss_kernel(erode(fake_B))
        loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
        loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))
        loss_G_B_ink = mse(ink_fake_B, torch.ones_like(ink_fake_B))

        # cycle loss
        cycle_B = gen_B(fake_A)
        cycle_A = gen_A(fake_B)
        cycle_B_loss = l1(B, cycle_B)
        cycle_A_loss = l1(A, cycle_A)

        # brush loss
        edge_real_A = torch.sigmoid(hed(A))
        edge_fake_B = torch.sigmoid(hed(fake_B))
        loss_edge = no_sigmoid_cross_entropy(edge_fake_B, edge_real_A)

        # add all together
        G_loss = (
            loss_G_B
            + loss_G_A
            + loss_G_B_ink * config.LAMBDA_INK
            + cycle_B_loss * config.LAMBDA_CYCLE
            + cycle_A_loss * config.LAMBDA_CYCLE
            + loss_edge * config.LAMBDA_BRUSH
        )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        loop.set_postfix(B_real=B_reals/(idx+1), B_fake=B_fakes/(idx+1))



def main():
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)
    disc_ink = Discriminator(in_channels=3).to(config.DEVICE)

    gen_A = Generator(img_channels=3).to(config.DEVICE)
    gen_B = Generator(img_channels=3).to(config.DEVICE)

    hed = HED().to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()) + list(disc_ink.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_A, disc_A, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_B, disc_B, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_INK, disc_ink, opt_disc, config.LEARNING_RATE,
        )

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

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train(disc_A, disc_B, disc_ink, gen_A, gen_B, hed, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_CRITIC_B)
            save_checkpoint(disc_ink, opt_disc, filename=config.CHECKPOINT_CRITIC_INK)


if __name__ == "__main__":
    main()
