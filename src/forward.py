import torch
import torch.nn.functional as F

from dataset import generate_dataset
from util import show_images


def linear_beta_schedule(steps, start=0.0001, end=0.02):
    return torch.linspace(start, end, steps)


def generate_noised_samples(
    x_0: torch.Tensor, steps: int, beta_start: float = 0.0001, beta_end: float = 0.02
):
    """Generate noised samples for diffusion model forward process.

    Params
    ------
    x_0: torch.Tensor, shape (0, 28, 28)

    Returns
    -------
    torch.Tensor, shape (step, 0, 28, 28)
        The tensor of the noised images.

    """

    noise = torch.randn_like(x_0)

    betas = linear_beta_schedule(steps=steps, start=0.0001, end=0.02)
    cummulative_betas = torch.cumsum(betas, axis=0)
    cummulative_alphas = 1 - cummulative_betas

    sqrt_cummulative_alphas = torch.sqrt(cummulative_alphas)
    sqrt_cummulative_betas = torch.sqrt(cummulative_betas)

    # TODO: codes for test.
    sqrt_cummulative_alphas = torch.ones_like(x_0) * sqrt_cummulative_alphas[-1]
    sqrt_cummulative_betas = torch.ones_like(noise) * sqrt_cummulative_betas[-1]

    return sqrt_cummulative_alphas * x_0 + sqrt_cummulative_betas * noise


if __name__ == "__main__":

    train_data, _ = generate_dataset()

    sample = next(iter(train_data))[0]

    output = generate_noised_samples(sample, steps=10)

    print(output.shape)
    show_images([sample, output])
