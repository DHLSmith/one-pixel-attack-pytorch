"""
Code for generating single pixel adversarial attacks.

Based on https://github.com/DebangLi/one-pixel-attack-pytorch/tree/master
"""

from pathlib import Path

import numpy as np

import argparse

import torch
import torch.nn.functional as F

import torchvision
from model_utilities.models.cifar_resnet import *
from model_utilities.training import modelfitting
from scipy.optimize import differential_evolution
from torch.utils.data import DataLoader
from torchbearer.magics import torchbearer

from model_utilities.training.modelfitting import get_device
from tqdm import tqdm

from single_pixel_attack import perturb_image  # gradient_importance.adverserial from git@github.com:feature-importance/model-utilities.git
from os import makedirs, path

def predict_classes(xs, img, target_class, net, minimize=True, device='cpu'):
    # perturbs each batch size copies of the img by setting /pixel/ pixels to new values determined by the tensor xs
    # then predicts on these images and return the softmax predictions for the target_class passed in
    imgs_perturbed = perturb_image(xs, img.clone())
    input = imgs_perturbed.to(device)
    predictions = F.softmax(net(input), dim=1).cpu().numpy()[:, target_class]

    return predictions if minimize else 1 - predictions


def attack_success(x, img, target_class, net, targeted_attack=False,
                   verbose=False, device='cpu'):
    attack_image = perturb_image(x, img.clone())
    input = attack_image.to(device)
    confidence = F.softmax(net(input), dim=1).cpu().numpy()[0]
    predicted_class = np.argmax(confidence)

    if verbose:
        print("Confidence: %.4f" % confidence[target_class])
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        return True


def attack(img, label, net, target=None, pixels=1, maxiter=75, popsize=400,
           verbose=False, device='cpu'):
    # img: 1*3*W*H tensor
    # label: a number

    targeted_attack = target is not None
    target_class = target if targeted_attack else label

    bounds = [(0, 32), (0, 32), (0, 255), (0, 255), (0, 255)] * pixels

    predict_fn = lambda xs: predict_classes(
        xs, img, target_class, net, target is None, device=device)
    callback_fn = lambda x, convergence: attack_success(
        x, img, target_class, net, targeted_attack, verbose, device=device)

    inits = np.zeros([popsize, len(bounds)])
    for init in inits:
        for i in range(pixels):
            init[i * 5 + 0] = np.random.random() * 32
            init[i * 5 + 1] = np.random.random() * 32
            init[i * 5 + 2] = np.random.normal(128, 127)
            init[i * 5 + 3] = np.random.normal(128, 127)
            init[i * 5 + 4] = np.random.normal(128, 127)

    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter,
                                           mutation=0.5,
                                           strategy="rand1bin",
                                           recombination=0,
                                           callback=callback_fn,
                                           polish=False,
                                           init=inits,
                                           updating='deferred',
                                           vectorized=True)

    attack_image = perturb_image(attack_result.x, img)
    attack_var = attack_image.to(device)
    predicted_probs = F.softmax(net(attack_var), dim=1).cpu().numpy()[0]

    predicted_class = np.argmax(predicted_probs)

    if (not targeted_attack and predicted_class != label) or (
            targeted_attack and predicted_class == target_class):
        return 1, attack_result.x.astype(int), predicted_class, label
    return 0, [None], predicted_class, label


def attack_all(net, loader, pixels=1, targeted=False, maxiter=75, popsize=400,
               verbose=False, device='cpu', samples=100):
    correct = 0
    success = 0
    success_rate = 0
    attack_data = {}

    with torch.no_grad():
        tloader = tqdm(loader)
        for batch_idx, (input, target) in enumerate(tloader):
            img_var = input.to(device)
            prior_probs = F.softmax(net(img_var), dim=1)
            _, indices = torch.max(prior_probs, 1)

            # only adverserial when the network was correct
            if target[0] != indices.cpu()[0]:
                continue

            correct += 1
            target = target.numpy()

            targets = [None] if not targeted else range(10)

            for target_class in targets:  
                # Note this executes once if targets is [None]
                # otherwise it will try each target_class other than the true label (target[0]) of the image
                if targeted:
                    if target_class == target[0]:
                        continue

                flag, x, predicted_class, label = attack(input, target[0], net,
                                                         target_class,
                                                         pixels=pixels,
                                                         maxiter=maxiter,
                                                         popsize=popsize,
                                                         verbose=verbose,
                                                         device=device)

                success += flag
                if targeted:
                    success_rate = float(success) / (9 * correct)
                else:
                    success_rate = float(success) / correct

                if flag == 1:
                    attack_data[batch_idx] = {"pixel": x,
                                              "attack_result_class": label,
                                              "pre_attack_class":
                                                  predicted_class,
                                              "true_class": target[0]}
                    tloader.set_postfix({"success_rate": success_rate})

            if correct == samples:
                break

    return success_rate, attack_data


def cc(name):
    if 'resnet' in name:
        return name.replace('resnet', 'ResNet')
    return name


def main():
    parser = argparse.ArgumentParser(
        description='One pixel adverserial with PyTorch')
    parser.add_argument('--model', default='resnet18_3x3',
                        help='The target model')
    parser.add_argument('--weights',
                        help='The model weights. Either a path to a file or '
                             'the name of the specific WeightsEnum for the '
                             'model', required=True)
    parser.add_argument('--pixels', default=1, type=int,
                        help='The number of pixels that can be perturbed in each image.')
    parser.add_argument('--maxiter', default=100, type=int,
                        help='The maximum number of iteration in the DE '
                             'algorithm.')
    parser.add_argument('--popsize', default=400, type=int,
                        help='The number of adversarial examples in each '
                             'iteration.')
    parser.add_argument('--samples', default=None, type=int,
                        help='The number of image samples to adverserial. Default '
                             'is to adverserial all images.')
    parser.add_argument('--targeted', action='store_true',
                        help='Set this switch to try to perturb each image to every incorrect class')
    parser.add_argument('--save', default='./results/results.csv',
                        help='Save location for the results.')
    parser.add_argument('--data', default=str(Path.home()) + "/data/",
                        help='Data location for storing image dataset')
    parser.add_argument('--verbose', action='store_true',
                        help='Print out additional information every '
                             'iteration.')

    args = parser.parse_args()
    makedirs(path.dirname(args.save), exist_ok=True)  # make the save directory if needed
    modelfitting.FORCE_MPS=True
    device = get_device('auto')

    print("==> Loading data and model...")
    transform_test = ResNet18_3x3_Weights.DEFAULT.transforms()
    test_set = torchvision.datasets.CIFAR10(root=args.data,
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    testloader = DataLoader(test_set, batch_size=1, shuffle=False,
                            num_workers=2)

    weights_enum = globals()[cc(args.model) + "_Weights"]
    if hasattr(weights_enum, args.weights):
        weights = weights_enum[args.weights]
        net = globals()[args.model](weights=weights).to(device)
    else:
        net = globals()[args.model]().to(device)
        weights = torch.load(args.weights, map_location=device,
                             weights_only=False)
        if torchbearer.MODEL in weights:
            weights = weights[torchbearer.MODEL]
        net.load_state_dict(weights)

    net.eval()

    print("==> Starting adverserial...")
    results, attack_data = attack_all(net, testloader,
                                      pixels=args.pixels,
                                      targeted=args.targeted,
                                      maxiter=args.maxiter,
                                      popsize=args.popsize,
                                      verbose=args.verbose,
                                      device=device,
                                      samples=args.samples)
    print("Final success rate: %.4f" % results)

    with open(args.save, 'w') as f:
        # Log index of image an pixel perturbation values for successful attacks
        # FIXME this only logs one of the pixels
        f.write("idx,row,col,r,g,b\n")
        for k, v in attack_data.items():
            v = v["pixel"]
            f.write(f"{k},{v[0]},{v[1]},{v[2]},{v[3]},{v[4]}\n")


if __name__ == '__main__':
    main()
