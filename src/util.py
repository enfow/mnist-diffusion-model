from typing import Any, List

import matplotlib.pyplot as plt
import torch

from dataset import generate_dataset


def show_images(imgs: List[torch.Tensor]):
    """Show list of images on a single row."""
    n_imgs = len(imgs)

    plt.figure(figsize=(n_imgs * 3, 3))

    for i, img in enumerate(imgs):
        plt.subplot(1, n_imgs, i + 1)
        plt.imshow(img.reshape(28, 28))

    plt.show()


if __name__ == "__main__":

    train_data, _ = generate_dataset()

    images = []
    for image, label in train_data:
        images.append(image)

        if len(images) >= 3:
            break

    show_images(images)
