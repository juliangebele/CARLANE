import json
import os
from PIL import Image


def balance_lanes(label_file, images_per_dir):
    """
    This script balances the bins of a recorded Town, if the number of images
    in their respective classes are not equally distributed.

    1. load labels_file (sim_train.json)
    2. count images per dir
    3. calc difference
    4. cut image paths from label file of source imgs
    5. mirror labels, change paths from e.g. left to right and change image name
    6. paste images in target dir
    """

    curve_counters = {
        'steep_left_curve/': 0,
        'left_curve/': 0,
        'straight/': 0,
        'right_curve/': 0,
        'steep_right_curve/': 0
    }

    print('Deleting old .png files...')
    for folder in os.listdir(root_dir):
        for label in os.listdir(root_dir+folder):
            if label.endswith('.png'):
                os.remove(os.path.join(root_dir+folder, label))

    # 1. open label file
    with open(label_file) as label_file:
        lines = [line for line in label_file.readlines()]

    new_label_file = open(new_file, "w")

    # 2. count images
    for curve_dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, curve_dir)):
            for file in os.listdir(os.path.join(root_dir, curve_dir)):
                if file.endswith('.jpg'):
                    curve_counters[curve_dir + '/'] += 1

    # 3. calc difference between folders and assign source and target
    curve_pair_list = [{}, {}]
    for curve in curve_counters:
        curve_counters[curve] = curve_counters[curve] - images_per_dir
        if 'steep' in curve:
            curve_pair_list[1][os.path.join(root_dir, curve)] = curve_counters[curve]
        elif 'straight' not in curve:
            curve_pair_list[0][os.path.join(root_dir, curve)] = curve_counters[curve]

    for i, curve_pair in enumerate(curve_pair_list):
        # if first item is the target, then swap the order
        if list(curve_pair.values())[0] < 0:
            swap = list(curve_pair.items())
            swap[0], swap[1] = swap[1], swap[0]
            curve_pair_list[i] = dict(swap)

    last_n_image_paths = [[], []]
    for i, curve_pair in enumerate(curve_pair_list):
        assert abs(list(curve_pair.values())[0]) == abs(list(curve_pair.values())[-1])

        # 4. cut image paths from source to target
        # get last n imagepaths of source
        counter = 0
        for line in lines[::-1]:
            curve_type = '/' + list(curve_pair.keys())[0].split('/')[-2] + '/'
            if curve_type in line and counter < list(curve_pair.values())[0]:
                counter += 1
                last_n_image_paths[i].append(line)

        for line in last_n_image_paths[i]:
            if line in lines:
                lines.remove(line)

    for line in lines:
        new_label_file.write(line)

    # reverse labels
    for path_list in last_n_image_paths:
        path_list.reverse()

    new_labels = [[], []]
    for i, curve_pair in enumerate(curve_pair_list):
        # get last file's name of target dir
        last_img = [file for file in os.listdir(list(curve_pair.keys())[1])][-1].split('.')[0]

        while last_img.startswith('0'):
            last_img = last_img[1:]
        last_img = int(last_img) + 1

        # 5. mirror labels, change paths from e.g. left to right and change image name
        for j, line in enumerate(last_n_image_paths[i]):
            line = json.loads(line)

            # mirror labels
            for lane in line['lanes']:
                for index, x in enumerate(lane):
                    if x != -2:
                        lane[index] = 1280 - x

            # swap lanes
            line['lanes'][0], line['lanes'][1] = line['lanes'][1], line['lanes'][0]

            # change paths
            if 'left' in line['raw_file']:
                line['raw_file'] = line['raw_file'].replace('left', 'right')
            elif 'right' in line['raw_file']:
                line['raw_file'] = line['raw_file'].replace('right', 'left')

            # change name
            old_name = line['raw_file'].split('/')[-1]
            line['raw_file'] = line['raw_file'].replace(old_name, f'{last_img + j:04d}.jpg')

            new_labels[i].append(json.dumps(line) + '\n')

    # save labels to file
    for label in new_labels:
        for line in label:
            new_label_file.write(line)

    for curve_pair in curve_pair_list:
        # 6. copy and paste images to target dir with new name
        print('Moving images from source to target')
        print(curve_pair)

        # get last file's name of target dir
        last_img = [file for file in os.listdir(list(curve_pair.keys())[1])][-1].split('.')[0]

        while last_img.startswith('0'):
            last_img = last_img[1:]
        last_img = int(last_img) + 1

        source_list = [os.path.join(list(curve_pair.keys())[0], img_path) for img_path in os.listdir(list(curve_pair.keys())[0])]

        # get last n elements to move
        source_list = source_list[-list(curve_pair.values())[0]:]

        for name, source in enumerate(source_list):
            target = list(curve_pair.keys())[1] + f'{last_img + name:04d}.jpg'

            # read and flip the image
            img = Image.open(source)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(target)
            os.remove(source)

    label_file.close()
    new_label_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--town_name", type=str, default="Town06", help="Name of the Town folder, whose images are balanced")
    parser.add_argument("--root_dir", type=str, default="./MoLane/data/train/sim/", help="root directory where the images are placed")
    parser.add_argument("--label_root_path", type=str, default="./MoLane/splits/", help="root directory where the labels are placed")
    parser.add_argument("--imgs_per_dir", type=int, default=1000, help="amount of images per folder")
    args = parser.parse_args()

    root_dir = os.path.join(args.root_dir, args.town_name)
    old_file = os.path.join(args.label_root_path, f'{args.town_name}.json')
    new_file = os.path.join(args.label_root_path, f'new_{args.town_name}.json')

    balance_lanes(old_file, args.imgs_per_dir)
