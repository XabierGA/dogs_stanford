import os
import numpy as np
import cv2
import random
from dog_clf_CNN import *


def parse_main_dir(main_dir_address):
    races_list = os.listdir(main_dir_address)
    nRaces = len(races_list)

    nImgs = 0
    for race in races_list:
        race_dir = os.listdir(main_dir_address + '/' + race)
        nImgs += len(race_dir)

    tensor = np.zeros((nImgs, pixels, pixels, rgb), dtype=np.float32)

    return tensor, nImgs, nRaces


def parse_img(img_route, dim):
    img = cv2.imread(img_route)
    dim = tuple(dim)
    img = cv2.resize(img, dim , interpolation = cv2.INTER_AREA)
    return img


def fill_tensor(main_dir_address, dim, verbose=0):
    races_list = os.listdir(main_dir_address)
    # Getting initialized tesor, number of images and number of different races
    tensor, nImgs, nRaces = parse_main_dir(main_dir_address)

    Y = []

    img_counter = 0

    for j,race in enumerate(races_list):
        race_name_ind = race.find("-")
        race_name = race[race_name_ind+1:] # +1 to ignore '-'
        if verbose:
            print("Parsing race \'"+race_name+"\' ("+str(j+1)+"/"+str(nRaces)+")")

        race_dir = os.listdir(main_dir_address + '/' + race)

        for i,img in enumerate(race_dir):
            tensor[img_counter] = parse_img(main_dir_address + '/' + race + '/' + img , dim)
            Y.append(race_name)

            img_counter += 1

    return tensor, Y, nRaces


def shuffle_dog(img_tensor , y_classes):

    c = list(zip(img_tensor, y_classes))
    random.shuffle(c)
    img_tensor, y_classes = zip(*c)

    return img_tensor , y_classes


if __name__ == "__main__":
    cwd = os.getcwd()
    images_dir = cwd+"/stanford-dogs-dataset/Images"
    print(images_dir)

    pixels = 50
    rgb = 3
    dim = (pixels, pixels)

    print(">>> Filling tensor...")
    IMG , Y, nClasses = fill_tensor(images_dir , dim, verbose=1)

    print(">>> Shuffling tensor...")
    IMG_shuff , Y_shuff = shuffle_dog(IMG , Y)

    print(">>> Compiling model...")
    model = CNN_dogs(128 , 4 , IMG_shuff , Y_shuff , nClasses)
    model.compile_model()
