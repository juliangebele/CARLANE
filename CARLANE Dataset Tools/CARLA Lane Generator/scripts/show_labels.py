import json
import os
import cv2


def get_lane_color(i):
    lane_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    return lane_colors[i % 5]


def input_images(data_root, input_file):
    """
    This method provides an easy way to visually validate train data by drawing labels on the frames and displaying them
    Args:
        input_file: labels file (filepath relative to data_root)
        data_root: path to dataset
    """
    with open(os.path.join(data_root, input_file)) as file:
        lines = file.readlines()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter(os.path.join(data_root, 'vid.avi'), fourcc, 30.0, (1280, 720))

    for line in lines:
        line = json.loads(line)
        image = cv2.imread(os.path.join(data_root, line['raw_file']))

        for i in range(len(line['lanes'])):
            lane = line['lanes'][i]
            for j in range(len(line['h_samples'])):
                if lane[j] is not -2:
                    cv2.circle(image, (lane[j], line['h_samples'][j]), 3, get_lane_color(i), -1)

        print(line['raw_file'])
        cv2.imshow('video', image)
        cv2.waitKey(20)
        vout.write(image)


if __name__ == '__main__':
    input_images('./MoLane/data/', 'Town03.json')
