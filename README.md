# One Pixel Attack with PyTorch

 This project is a simple PyTorch implementation of ["One pixel attack for fooling deep neural networks"](https://arxiv.org/abs/1710.08864) on the [Cifar10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The code is developed upon [Pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) and [one-pixel-attack-keras](https://github.com/Hyperparticle/one-pixel-attack-keras).

## Results
|model   |Accuracy on the test set| Success Rate (1 pixel, untargeted)|Success Rate (3 pixels, untargeted)|
|--------|------------------------|-----------------------------------|-----------------------------------|
|vgg16   |     93.42%             |              ~50.0%                |         ~93.0%           |
|res18   |     94.94%             |              ~27.7%               |         ~78.0%           |
|res101  |     94.51%             |              ~19.0%               |         ~63.3%                  |               


# Example Usage
$ python attack.py --weights CIFAR10_s0

or, to run a fairly short job:
'''            "args": ["--weights", "CIFAR10_s0",
                        "--pixels", "1", // 
                        "--maxiter", "100",  // limit the evolution
                        "--popsize", "400",   // number of interventions per image?
                        "--samples", "10",  // number of images to try to perturb
                        //"--targeted", 
                        //"--save", "",
                        //[--data DATA]
                        "--verbose"
'''                        

# Other files
train.py is used to train a model on CIFAR10 images if needed.

# Changes Needed
1. log more than the first pixel changed if adjusting several
1. improve logging to ensure perturbations can be read in and used for another experiment
1. consider refactoring so that images can be perturbed using logged results read in and known seeds, datasets, model, weights etc