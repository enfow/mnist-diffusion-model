from typing import Tuple

import torch
import torch.nn.functional as F

from dataset import generate_dataloader
from util import show_images


def linear_beta_schedule(steps, start=0.0001, end=0.02):
    return torch.linspace(start, end, steps)


def generate_noised_samples(
    x_0: torch.Tensor,
    step: int,
    total_steps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate noised samples for diffusion model forward process.

    Params
    ------
    x_0: torch.Tensor, shape (BATCH_SIZE, 0, 28, 28)
        original image
    steps: int
        number of noising steps

    Returns
    -------
    torch.Tensor, shape (step, 0, 28, 28)
        The tensor of the noised images.

    Notes
    -----
    How to add noise
        x_t = sqrt(cummulative_alpha) * x_0 + sqrt(1 - cummulative_alpha) * noise
        using reparameterization trick.

    closed form for forward process
        q(x_t | x_0) = N(
            x_t;
            sqrt(\bar{alpha})x_0,  # mean
            (1 - \bar{alpha_t})I   # variance
        )
        where
            alpha_t = 1 - beta_t
            \bar = \Pi_{x=1}^t \alpha_s

        see https://arxiv.org/abs/2006.11239
    """

    noise = torch.randn_like(x_0)

    betas = linear_beta_schedule(steps=total_steps, start=0.0001, end=0.02)
    cummulative_betas = torch.cumsum(betas, axis=0)
    cummulative_alphas = 1 - cummulative_betas

    sqrt_cummulative_alphas = torch.sqrt(cummulative_alphas)
    sqrt_cummulative_betas = torch.sqrt(cummulative_betas)

    # TODO: codes for test.
    sqrt_cummulative_alphas = torch.ones_like(x_0) * sqrt_cummulative_alphas[step]
    sqrt_cummulative_betas = torch.ones_like(noise) * sqrt_cummulative_betas[step]

    return sqrt_cummulative_alphas * x_0 + sqrt_cummulative_betas * noise, noise


if __name__ == "__main__":

    train_dataloader, _ = generate_dataloader()

    batch_sample = next(iter(train_dataloader))[0]

    images = []
    for step in range(0, 10):
        output, noise = generate_noised_samples(batch_sample, step=step, total_steps=50)
        images.append(output[0])
    show_images(images)
