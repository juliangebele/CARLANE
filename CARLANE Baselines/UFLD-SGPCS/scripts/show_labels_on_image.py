import json
import os
import cv2


def get_lane_color(i):
    lane_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    return lane_colors[i % 5]


def input_images(input_file, data_root):
    """ This method provides an easy way to visually validate train data by drawing labels on the frames and displaying them

    Args:
        input_file: labels file (filepath relative to data_root)
        data_root: path to dataset
    """
    with open(os.path.join(data_root, input_file)) as file:
        lines = file.readlines()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # vout = cv2.VideoWriter(os.path.join(data_root, 'vid.avi'), fourcc, 30.0, (1280, 720))

    for line in lines:
        dict = json.loads(line)

        image = cv2.imread(os.path.join(data_root, dict['raw_file']))

        for i in range(len(dict['lanes'])):
            lane = dict['lanes'][i]
            for j in range(len(dict['h_samples'])):
                # if lane[j] is not -2:
                    cv2.circle(image, (lane[j], dict['h_samples'][j]), 5, get_lane_color(i), -1)

        print(dict['raw_file'])
        cv2.imshow('video', image)
        cv2.waitKey(10)
        # vout.write(image)


if __name__ == '__main__':
    input_images('train_gt.json', '/path/of/data_root/')
