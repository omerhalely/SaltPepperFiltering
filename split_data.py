import os
import numpy as np


def split_data(data_path, train_percentage):
    images_list = os.listdir(data_path)
    number_of_train_images = int(train_percentage * len(images_list))

    array_size = number_of_train_images
    min_value = 0
    max_value = len(images_list)

    train_indexes = np.random.choice(np.arange(min_value, max_value), size=array_size, replace=False).tolist()
    train_file = open("./Train_images.log", "w")
    for i in range(len(train_indexes) - 1):
        train_file.write(str(train_indexes[i]) + " ")
    train_file.write(str(train_indexes[-1]))
    train_file.close()


if __name__ == "__main__":
    data_path = "./data"
    train_percentage = 0.8
    split_data(data_path=data_path, train_percentage=train_percentage)