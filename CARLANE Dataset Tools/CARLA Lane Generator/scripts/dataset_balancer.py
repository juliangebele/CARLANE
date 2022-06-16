import json
import os
import numpy as np
import math
import cv2
from PIL import Image


def balance_lanes(label_file):
    """
    Balance the images and labels from a town, which contains not equally distributed images per bin (e.g. Town04, Town06)
    First execute create_seg_labels_and_index_files
    1. load labels_file (train_gt.txt)
    2. count images per dir
    3. calc diff
    4. copy images from source dir
    5. mirror images and labels
    6. paste in target dir
    7. change pathnames in labels file
    """
    images_per_dir = 50
    root_dir = './MoLane/Town10HD/'
    curve_counters = {
        'steep_left_curve/': 0,
        'left_curve/': 0,
        'straight/': 0,
        'right_curve/': 0,
        'steep_right_curve/': 0
    }

    source_dir = {}
    target_dir = {}

    # open label file
    with open(label_file) as label_file:
        lines = [line for line in label_file.readlines()]

    # count images
    for curve_dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, curve_dir)):
            for file in os.listdir(os.path.join(root_dir, curve_dir)):
                if file.endswith('.jpg'):
                    curve_counters[curve_dir + '/'] += 1

    # calc difference between folders and assign source and target
    for curve in curve_counters:
        curve_counters[curve] = curve_counters[curve] - images_per_dir
        if curve_counters[curve] > 0:
            source_dir = {os.path.join(root_dir, curve): curve_counters[curve]}
        if curve_counters[curve] < 0:
            target_dir = {os.path.join(root_dir, curve): curve_counters[curve]}

    assert abs(list(source_dir.values())[0]) == abs(list(target_dir.values())[0])
    print('source:', source_dir, 'target:', target_dir)

    # get last n imagepaths of source
    image_source = [file for file in os.listdir(list(source_dir.keys())[0])]
    image_source = image_source[-list(source_dir.values())[0]:]

    # get last file's name
    image_target = [file for file in os.listdir(list(target_dir.keys())[0])][-1].split('.')[0]
    while image_target.startswith('0'):
        image_target = image_target[1:]
    image_target = int(image_target)

    for file in image_source:
        image_target += 1
        source = os.path.join(list(source_dir.keys())[0], file)
        target = os.path.join(list(target_dir.keys())[0], f'{image_target}.jpg')
        print(source)
        print(target)

        # read and flip the image
        img = Image.open(source)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img.save(source)

        # move the files from source to target
        # os.rename(source, target)

        # change the path of the label file
        suffix_source = os.path.join(*source.split('/')[-3:])
        print(suffix_source[:-4])

        for i, line in enumerate(lines):
            if suffix_source in line:
                print(i)
                # cut line and paste it at the end with inversed direction

    print(lines[0])

    label_file.close()


if __name__ == '__main__':
    balance_lanes('./MoLane/data/splits/sim_train.txt')
