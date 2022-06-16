import os
import json
import random
import shutil


def sample_real_test_val_tusimple_data():
    """
    Sampling strategy for real_val and real_test from tusimple:
    val: sample 1642 from 2782 (test set) and append to 358 validation
    test: sample 1000 from remaining 1140 (discard remaining 140 samples)
    """

    real_val_file_name = args.real_val_file_name
    real_test_file_name = args.real_test_file_name
    new_real_val_file_name = args.new_real_val_file_name
    new_real_test_file_name = args.new_real_test_file_name

    new_real_val_file = open(new_real_val_file_name, 'w')
    new_real_test_file = open(new_real_test_file_name, 'w')

    with open(real_val_file_name, 'r') as real_val_file:
        real_val_lines = [line for line in real_val_file.readlines()]

    with open(real_test_file_name, 'r') as real_test_file:
        real_test_lines = [line for line in real_test_file.readlines()]
        shuffle_real_val_lines = real_test_lines
        tmp_real_test_lines = real_test_lines.copy()

        random.shuffle(shuffle_real_val_lines)
        shuffle_real_val_lines = shuffle_real_val_lines[:1642]

    for line in real_test_lines:
        if line in shuffle_real_val_lines:
            real_val_lines.append(line)
            tmp_real_test_lines.remove(line)

    for line in real_val_lines:
        new_real_val_file.write(line)

    samples_to_remove = len(tmp_real_test_lines) - 1000
    for i in range(samples_to_remove):
        tmp_real_test_lines.remove(random.choice(tmp_real_test_lines))

    for line in tmp_real_test_lines:
        new_real_test_file.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_val_file_name", type=str, default=".TuSimple/label_data_0531.json", help="TuSimple's validation labels")
    parser.add_argument("--real_test_file_name", type=str, default="./TuSimple/test_label.json", help="TuSimple's test labels")
    parser.add_argument("--new_real_val_file_name", type=str, default="./MuLane/new_real_val_tusimple.json", help="new validation file sampled from TuSimple")
    parser.add_argument("--new_real_test_file_name", type=str, default="./MuLane/new_real_test_tusimple.json", help="new test file sampled from TuSimple")

    args = parser.parse_args()

    sample_real_test_val_tusimple_data()
