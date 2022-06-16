import os
import json
import random
import shutil


def sample_simulation_data():
    """
    Sampling strategy for sim_train:
    sample 24k from 80 k
    10 towns and 5 bins per town
    6 towns with 60k imgs (75%), 4 towns with 20k imgs (25%)
    from 24k imgs: 18k (=75%) 6k (=25%)
    18k imgs / 6 towns = 3k per town, 6k imgs / 4 towns = 1.5k per town
    3k / 5 bins = 600 imgs per bin (600 imgs * 5 bins * 6 towns = 18k imgs)
    1.5k / 5 bins = 300 imgs per bin (300 imgs * 5 bins * 4 towns = 6k imgs)

    Sampling strategy for sim_val:
    60 imgs per bin (60 imgs * 5 bins * 6 towns = 1800 imgs)
    30 imgs per bin (30 imgs * 5 bins * 4 towns = 600 imgs)
    """

    sim_labels = args.sim_labels
    root_dir = args.root_dir
    root = args.root

    target_dir_root = args.target_dir_root
    new_label_file_name = args.new_label_file_name

#########################################################################################

    with open(sim_labels, 'r') as label_file:
        sim_lines = [line for line in label_file.readlines()]
        sim_train_json_lines = [json.loads(line)['raw_file'] for line in sim_lines]

    new_label_file = open(new_label_file_name, 'w')

    # sample n images from D18 bins
    print("Sampling images from Source dataset")
    dataset_list = []
    for town in os.listdir(root_dir):
        if 'sim_train' in sim_labels:
            n = 300 if '04' in town or '06' in town else 600
        else:
            n = 30 if '04' in town or '06' in town else 60
        town_name = root_dir + '/' + town
        # print(town_name)
        for bin in os.listdir(town_name):
            bin_image_list = []
            bin_name = town_name + '/' + bin
            for image in os.listdir(bin_name):
                if image.endswith('.jpg'):
                    img_name = bin_name + '/' + image
                    bin_image_list.append(img_name)

            random.shuffle(bin_image_list)
            bin_image_list = bin_image_list[:n]
            bin_image_list.sort()
            dataset_list += bin_image_list

            # print(bin_image_list)

    # generate new list with sampled lines
    print("Copying images to new dataset")
    index_list = []
    for tmp_line in dataset_list:
        if 'sim_train' in sim_labels:
            prefix = 'train/sim'
        else:
            prefix = 'val/sim'
        postfix = prefix + tmp_line.split('sim')[1]  # prefix + /Town03/left_curve/0003.jpg
        index = sim_train_json_lines.index(postfix)
        index_list.append(index)

        source_image = os.path.join(root, postfix)
        tgt_path = postfix.split('/')
        target_dir = os.path.join(target_dir_root, tgt_path[0] + '/' + tgt_path[1] + '/' + tgt_path[2] + '/' + tgt_path[3])

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        shutil.copy(source_image, target_dir)

    print("Creating new_sim_val.json")
    for i in index_list:
        new_label_file.write(sim_lines[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_labels", type=str, default="./MoLane/splits/sim_val.json", help="path to simulation labels")
    parser.add_argument("--root_dir", type=str, default="./MoLane/data/val/sim", help="root directory where the images are placed")
    parser.add_argument("--root", type=str, default="./MoLane", help="root directory of the source dataset")
    parser.add_argument("--target_dir_root", type=str, default="./MuLane/", help="root directory of the target dataset")
    parser.add_argument("--new_label_file_name", type=str, default="./MuLane/data/new_sim_val.json", help="name of the new label file")

    args = parser.parse_args()

    sample_simulation_data()
