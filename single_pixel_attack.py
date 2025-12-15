import numpy as np

from gradient_importance.utils.utils import CIFAR10_MEAN, CIFAR10_STD


def perturb_image(xs, img):
    if xs.ndim < 2:
        xs = np.array([xs]).T

    batch = xs.shape[1]
    imgs = img.repeat(batch, 1, 1, 1)
    xs = xs.astype(int)

    count = 0
    for x in xs.T:
        pixels = np.split(x, len(x) // 5)

        for pixel in pixels:
            row, col, r, g, b = pixel
            imgs[count, 0, row, col] = ((r / 255.0 - CIFAR10_MEAN[0]) /
                                        CIFAR10_STD[0])
            imgs[count, 1, row, col] = ((g / 255.0 - CIFAR10_MEAN[1]) /
                                        CIFAR10_STD[1])
            imgs[count, 2, row, col] = ((b / 255.0 - CIFAR10_MEAN[2]) /
                                        CIFAR10_STD[2])
        count += 1

    return imgs
