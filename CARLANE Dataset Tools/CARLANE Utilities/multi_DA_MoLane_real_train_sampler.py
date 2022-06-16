import os
import json
import random
import shutil


def sample_real_train_data():
    """
    Sampling strategy for real_train:
    sample 3268 from 43843
    16 bins in total, 102 imgs per bin
    16 * 204 = 3264
    sample 1 imgs randomly from straight, s_track, round_black and round_gray
    """

    root_dir = args.root_dir
    root = args.root

    target_dir_root = args.target_dir_root
    real_train_file_name = args.real_train_file_name

    n_samples = args.n_samples

    #########################################################################################

    real_train_file = open(real_train_file_name, 'w')

    def load_images(image_dir):
        """
        Load images of a folder structure recursively.
        """
        for filename in os.listdir(image_dir):
            if filename.endswith('.jpg'):
                real_train_image_list.append(image_dir)
            elif not filename.endswith('.png'):
                load_images(image_dir + filename + '/')

    real_train_image_list = []
    load_images(root_dir)
    real_train_image_list = list(dict.fromkeys(real_train_image_list))

    for _ in range(4):
        category_list = [real_train_image_list[:4]]
        del real_train_image_list[:4]
        real_train_image_list += category_list

    for category in real_train_image_list:
        additional_bin_sample = random.choice(category)
        for img_path in category:
            if additional_bin_sample == img_path:
                n = n_samples + 1
            else:
                n = n_samples
            img_list = []
            for img in os.listdir(img_path):
                path = img_path.split('/')[4:]
                img_list.append('/'.join(path) + img)

            random.shuffle(img_list)
            img_list = img_list[:n]
            img_list.sort()
            # print(img_list)

            for im_path in img_list:
                real_train_file.write(im_path + '\n')
                target_image = os.path.join(target_dir_root, im_path)
                target_dir = target_image.split('/')[:-1]
                target_dir = '/'.join(target_dir)
                source_image = os.path.join(root, im_path)

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                shutil.copy(source_image, target_dir)

    real_train_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./MoLane/data/train/real/", help="root directory where the images are placed")
    parser.add_argument("--root", type=str, default="./MoLane", help="root directory of the source dataset")
    parser.add_argument("--target_dir_root", type=str, default="./MuLane/", help="root directory of the target dataset")
    parser.add_argument("--real_train_file_name", type=str, default="./MuLane/splits/real_train.txt", help="name of the new label file")
    parser.add_argument("--n_samples", type=int, default=204, help="number of images per folder to sample")

    args = parser.parse_args()

    sample_real_train_data()
