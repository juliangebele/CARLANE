import json
import time
import os
import cv2
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot


def get_lane_color(i):
    # lane_colors = [(210, 116, 25), (210, 116, 25), (210, 116, 25), (210, 116, 25), (255, 0, 255)]
    lane_colors = [(94, 102, 255), (94, 102, 255), (94, 102, 255), (94, 102, 255), (255, 0, 255)]
    return lane_colors[i % len(lane_colors)]


def visualize_labels(data_root, input_file, video_name, draw_points=False):
    """
    This method provides an easy way to visually validate image data by drawing labels on the
    frames and displaying them. Place this file together with the label file in the same root
    directory, make sure the .json labels and train or val folder containing the image data
    are on the same level.

    Args:
        data_root: path to dataset
        input_file: labels file (filepath relative to data_root)
        video_name: name of the video being created
        draw_points: draw points or lines
    """
    with open(os.path.join(data_root, input_file)) as file:
        lines = file.readlines()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter(os.path.join(data_root, video_name), fourcc, 30.0, (1280, 720))

    for line in lines:
        line = json.loads(line)
        image = cv2.imread(os.path.join(data_root, line['raw_file']))

        if draw_points:
            for i in range(len(line['lanes'])):
                lane = line['lanes'][i]
                for j in range(len(line['h_samples'])):
                    if lane[j] is not -2:
                        point = (lane[j], line['h_samples'][j])
                        cv2.circle(image, point, 3, get_lane_color(i), -1)
        else:
            pred_lanes_vis = [[(x, y) for (x, y) in zip(lane, line['h_samples']) if x >= 0] for lane in line['lanes']]
            for i, lane in enumerate(pred_lanes_vis):
                cv2.polylines(image, np.int32([lane]), isClosed=False,  color=get_lane_color(i), thickness=10)

        print(line['raw_file'])
        cv2.imshow('video', image)
        cv2.waitKey(20)
        cv2.imwrite(f'./save_path/images/frame_{int(time.time() * 1000000)}.jpg', image)
        vout.write(image)


def visualize_new_labels(data_root, input_file, video_name, draw_points=False, fit_curve=False):
    """
    This method provides an easy way to visually validate image data by drawing labels on the
    frames and displaying them. Place this file together with the label file in the same root
    directory, make sure the .json labels and train or val folder containing the image data
    are on the same level.

    Args:
        data_root: path to dataset
        input_file: labels file (filepath relative to data_root)
        video_name: name of the video being created
        draw_points: draw points or lines
        fit_curve: curve_fitting with polynomial function
    """
    with open(os.path.join(data_root, input_file)) as file:
        lines = file.readlines()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter(os.path.join(data_root, video_name), fourcc, 30.0, (1280, 720))

    img_list = []
    for ln in os.listdir('./save_path/images/'):
        img_list.append(ln)

    for ind, line in enumerate(lines):
        line = json.loads(line)
        image = cv2.imread(os.path.join('./save_path/images/', img_list[ind]))

        if fit_curve:
            for lane in line["lanes"]:
                if len(list(filter(lambda x: (x != -2), lane))) < 4:
                    continue
                curve_points = fitting_curve(lane, line['h_samples'])
                for point in curve_points:
                    cv2.circle(image, point, 3, (0, 0, 255), -1)

        if draw_points:
            for i in range(len(line['lanes'])):
                lane = line['lanes'][i]
                for j in range(len(line['h_samples'])):
                    if lane[j] is not -2:
                        point = (lane[j], line['h_samples'][j])
                        cv2.circle(image, point, 3, get_lane_color(i), -1)
        else:
            pred_lanes_vis = [[(x, y) for (x, y) in zip(lane, line['h_samples']) if x >= 0] for lane in line['lanes']]
            for i, lane in enumerate(pred_lanes_vis):
                cv2.polylines(image, np.int32([lane]), isClosed=False,  color=get_lane_color(i), thickness=10)

        print(line['raw_file'])
        cv2.imshow('video', image)
        cv2.waitKey(20)
        # cv2.imwrite(f'./save_path/images/frame_{int(time.time() * 1000000)}.jpg', image)
        vout.write(image)


if __name__ == '__main__':
    # visualize_labels('./MuLane/data/', 'target_test.json', 'video.avi')
    visualize_new_labels('./MuLane/data/', 'UFLD_SO_RN34.json', 'UFLD_SO.avi')
