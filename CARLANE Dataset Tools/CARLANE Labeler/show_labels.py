import json
import os
import cv2


def get_lane_color(i):
    lane_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    return lane_colors[i % 5]


def input_images(input_file, data_root):
    """
    This method provides an easy way to visually validate train data by drawing labels on the frames and displaying them
    Args:
        input_file: labels file (filepath relative to data_root)
        data_root: path to dataset
    """
    with open(os.path.join(data_root, input_file)) as file:
        lines = file.readlines()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter(os.path.join(data_root, 'real_test.avi'), fourcc, 30.0, (1280, 720))

    for line in lines:
        dict = json.loads(line)

        image = cv2.imread(os.path.join(data_root, dict['raw_file']))

        # draw points
        # for i in range(len(dict['lanes'])):
        #     lane = dict['lanes'][i]
        #     for j in range(len(dict['h_samples'])):
        #         if lane[j] is not -2:
        #             cv2.circle(image, (lane[j], dict['h_samples'][j]), 3, get_lane_color(i), -1)

        # draw lines
        for i in range(len(dict['lanes'])):
            for j in range(len(dict['h_samples']) - 1):
                if dict['lanes'][i][j] != -2 and dict['lanes'][i][j+1] != -2:
                    cv2.line(image, (dict['lanes'][i][j], dict['h_samples'][j]), (dict['lanes'][i][j+1], dict['h_samples'][j+1]), get_lane_color(i), 3)

        print(dict['raw_file'])
        cv2.imshow('video', image)
        cv2.waitKey(2000)
        vout.write(image)


if __name__ == '__main__':
    input_images('label_file.json', 'data_root/')
