# from ProposedMethod.QueryEfficient.Scratch import Attack
from MOAA.MOAA import Attack
from Cifar10Models import Cifar10Model # Can be changes to ImageNetModels
from LossFunctions import UnTargeted, Targeted
import numpy as np
import argparse
import os
import torch
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt


def get_images_and_labels():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Tạo nhãn mục tiêu khác với nhãn đúng
    y_target = (y_test.flatten() + 1) % 10  # Nhãn mục tiêu là (y_test + 1) % 10
    return x_test, y_test.flatten(), y_target


if __name__ == "__main__":
    """
    Non-Targeted
    pc = 0.1
    pm = 0.4
    
    Targeted:
    pc = 0.1
    pm = 0.2
    """
    np.random.seed(0)

    pc = 0.1
    pm = 0.4

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="0 or 1", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--save_directory", type=int)
    args = parser.parse_args()

    x_test, y_test, y_target = get_images_and_labels()
    
    # labels.
    model = Cifar10Model(0)


    for i in range (0, 100):
        # loss = Targeted(model, y_test[i], y_target[i], to_pytorch=True)
        # img_ = torch.from_numpy(x_test[i]).permute(2, 0, 1)
        # img_ = img_[None, :]
        # preds = model.predict(img_).flatten()
        # y = int(torch.argmax(preds))
        print(f'Attack image {i + 1} => Label: {y_test[i]}')

        loss = UnTargeted(model, y_target[i], to_pytorch=True) # to_pytorch is True only is the model is a pytorch model
        params = {
            "x": x_test[i], # Image is assume to be numpy array of shape height * width * 3
            "eps": 500, # number of changed pixels
            "iterations": 500 // 2, # model query budget / population size
            "pc": pc, # crossover parameter
            "pm": pm, # mutation parameter
            "pop_size": 2, # population size
            "zero_probability": 0.3,
            "include_dist": True, # Set false to not consider minimizing perturbation size
            "max_dist": 1e-5, # l2 distance from the original image you are willing to end the attack
            "p_size": 2, # Perturbation values have {-p_size, p_size, 0}. Change this if you want smaller perturbations.
            "tournament_size": 2, #Number of parents compared to generate new solutions, cannot be larger than the population
            # "save_directory": args.save_directory
            "save_directory": f'Result2/result_targeted_{i}'
        }
        attack = Attack(params)
        attack.attack(loss)
