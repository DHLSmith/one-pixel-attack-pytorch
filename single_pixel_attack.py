import numpy as np

# From git@github.com:feature-importance/model-utilities.git
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

def perturb_image(xs, img):
    # xs is the set of pixel perturabations to add
    if xs.ndim < 2:
        xs = np.array([xs]).T

    batch = xs.shape[1]
    imgs = img.repeat(batch, 1, 1, 1)
    # imgs is popsize batch x 3,32,32 - i.e. duplicates of the img being tested

    xs = xs.astype(int)
    # xs is 5 lists of batch-size entries (wider if more pixels being perturbed)
    count = 0
    for x in xs.T:
        pixels = np.split(x, len(x) // 5)
        # perturb each image in the batch by overwriting a pixel at row,col with a specific rgb value
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
