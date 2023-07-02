import time
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import json


class DataLoader(Dataset):
    def __init__(self, data_path, batch_size, train_percent, resize, p, data_type="train"):
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_percent = train_percent
        self.resize = resize
        self.p = p
        self.data_type = data_type
        self.images = self.load_data()

    def get_number_of_images(self):
        animals_dir = os.listdir(self.data_path)
        counter = 0
        for i in range(len(animals_dir)):
            animal_dir = os.path.join(self.data_path, animals_dir[i])
            counter += len(os.listdir(animal_dir))
        return counter

    def load_data(self):
        all_images = os.listdir(self.data_path)
        images = []
        train_images = open("./Train_images.log", "r")
        images_indexes = train_images.readline().split()
        for i in range(len(images_indexes)):
            images_indexes[i] = int(images_indexes[i])
        for i in range(len(all_images)):
            if (i in images_indexes and self.data_type == "train") or (i not in images_indexes and self.data_type == "test"):
                filename = os.path.join(self.data_path, all_images[i])
                image = plt.imread(filename)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = cv2.resize(image, self.resize)
                images.append(image)

        images = torch.from_numpy(np.array(images))
        images = torch.split(images, self.batch_size)
        if images[-1].size(0) != self.batch_size:
            images = images[:-1]
        return images

    def apply_noise(self, image):
        mask = torch.rand(image.shape)
        salt_mask = mask < self.p / 2
        pepper_mask = mask < self.p
        noisy_tensor = image.clone()
        noisy_tensor[pepper_mask] = 0
        noisy_tensor[salt_mask] = 255
        return noisy_tensor

    def __len__(self):
        if self.data_type == "train":
            return int(len(os.listdir(self.data_path)) * self.train_percent / self.batch_size)
        if self.data_type == "test":
            return int(len(os.listdir(self.data_path)) * (1 - self.train_percent) / self.batch_size)

    def __getitem__(self, idx):
        noisy_image = self.apply_noise(self.images[idx]).float()
        noisy_image = torch.unsqueeze(noisy_image, dim=1)
        image = self.images[idx].float()
        image = torch.unsqueeze(image, dim=1)
        return noisy_image, image


if __name__ == "__main__":
    data_path = "./data"
    batch_size = 10
    train_percent = 0.8
    data_type = "train"
    resize = (630 // 2, 530 // 2)
    p = 0.05
    data = DataLoader(data_path=data_path, batch_size=batch_size, train_percent=train_percent, resize=resize,
                      p=p, data_type=data_type)
    print(len(data))
    noisy_images, images = data[0]
    first_noisy_image = noisy_images[0].numpy()
    plt.imshow(first_noisy_image, cmap="gray")
    plt.show()
