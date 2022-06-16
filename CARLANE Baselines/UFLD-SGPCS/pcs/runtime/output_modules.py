import os
import time
import json
import datetime
import cv2
import numpy as np
from scipy import special
from pcs.eval.lane import LaneEval
import torch
import torch.nn.functional as F


def evaluate_predictions(output):
    """
    Tries to improve the estimation by including all probabilities instead of only using the most probable class
    Args:
        output: one result sample

    Returns:
        2D array containing x values (float) per h_sample and lane
    """
    # Process on CPU with numpy
    # output = output.data.cpu().numpy()
    # out_loc = np.argmax(output, axis=0)  # get most probable x-class per lane and h_sample
    # prob = special.softmax(output[:-1, :, :], axis=0)
    # idx = np.arange(100)
    # idx = idx.reshape(-1, 1, 1)
    # loc = np.sum(prob * idx, axis=0)
    # loc[out_loc == 100] = -2
    # out_loc = loc

    # Process on GPU with pytorch
    out_loc = torch.argmax(output, dim=0)  # get most probable x-class per lane and h_sample
    prob = F.softmax(output[:-1, :, :], dim=0)
    idx = torch.arange(100).cuda()
    idx = idx.reshape(-1, 1, 1)
    loc = torch.sum(prob * idx, dim=0)
    loc[out_loc == 100] = -2
    out_loc = loc

    # Map x-axis (griding_num) estimations to image coordinates
    lanes = []
    offset = 0.5  # different values used in ufld project. demo: 0.0, test: 0.5
    for i in range(out_loc.shape[1]):
        out_i = out_loc[:, i]
        lane = [int(torch.round((loc + offset) * 1280.0 / (100 - 1))) if loc != -2 else -2 for loc in out_i]
        # lane = [int((loc + offset) * float(1280) / (100 - 1)) if loc != -2 else -2 for loc in out_i]
        lanes.append(lane)
    return lanes


def get_filename_date_string():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def get_lane_color(i):
    lane_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    return lane_colors[i % len(lane_colors)]


class VisualOut:
    """
    Provides different visual output types

    * show live video
    * save as video to disk
    * save as image files to disk
    * visualization as lines instead of dots
    """

    def __init__(self, config):
        self.img_width = config.data_params.image_width
        self.img_height = config.data_params.image_height
        self.griding_num = config.model_params.griding_num
        self.out_dir = config.output_params.out_dir
        self.data_root = config.data_params.data_root
        self.scaled_h_samples = config.data_params.scaled_h_samples
        self.cls_num_per_lane = len(self.scaled_h_samples)
        self.enable_live_video = config.output_params.enable_live_video
        self.enable_video_export = config.output_params.enable_video_export
        self.enable_image_export = config.output_params.enable_image_export
        self.enable_line_mode = config.output_params.enable_line_mode

        if self.enable_video_export:
            # init video out
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            model_name = config.test_params.trained_model.split("/")[-1]
            out_full_path = os.path.join(self.out_dir, f'{get_filename_date_string()}_{model_name}.avi')
            self.vout = cv2.VideoWriter(out_full_path, fourcc, 30.0, (self.img_width, self.img_height))

    def out(self, output, names, frames):
        """
        Generate visual output

        Args:
            output: network result (list of samples containing probabilities per sample)
            names: filenames for output, if empty: frames have to be provided
            frames: source frames, if empty: names have to be provided
        """
        if not names and not frames:
            raise Exception('frames or names have to be provided')

        for i in range(len(output)):
            if frames:
                vis = frames[i]
            else:
                vis = cv2.imread(os.path.join(self.data_root, names[i]))
            if vis is None:
                raise Exception('failed to load frame')

            lanes = np.array(evaluate_predictions(output[i]))  # get x coordinates based on probabilities and scale the output
            for j in range(lanes.shape[0]):  # iterate over lanes
                lane = lanes[j, :]
                if np.sum(lane != -2) > 2:  # If more than two points found for this lane
                    color = get_lane_color(j)
                    for k in range(lanes.shape[1]):
                        img_x = lane[k]
                        img_y = self.scaled_h_samples[k]
                        if img_x != -2:
                            if self.enable_line_mode:
                                # find all previous points for current lane (in reverse order) that are not -2
                                # and store indexes of these points in prev_points
                                prev_points = [x for x in range(k - 1, -1, -1) if lane[x] != -2]
                                if prev_points:
                                    cv2.line(vis, (lane[prev_points[0]], self.scaled_h_samples[prev_points[0]]), (img_x, img_y), color, 10)
                            else:
                                cv2.circle(vis, (img_x, img_y), 5, color, -1)
            if self.enable_live_video:
                cv2.imshow('video', vis)
                cv2.waitKey(1)
            if self.enable_video_export:
                self.vout.write(vis)
            if self.enable_image_export:
                out_path = os.path.join(self.out_dir, names[i] if names else f'frame_{int(time.time() * 1000000)}.jpg')
                cv2.imwrite(out_path, vis)


class JsonOut:
    """
    Provides the ability to output detected data in a json like format (one json object per line) to a file
    This file will be analog to the source labels you are using for training
    """

    def __init__(self, config):
        self.scaled_h_samples = config.data_params.scaled_h_samples
        self.out_dir = config.output_params.out_dir
        self.out_file = open(os.path.join(self.out_dir, f'{get_filename_date_string()}.json'), 'w')

    def out(self, output, names, frames):
        """
        Generate json output to text file

        Args:
            output: network result (list of samples)
            names: filenames for output
        """
        # iterate over samples
        for i in range(len(output)):
            lanes = evaluate_predictions(output[i])  # get x coordinates based on probabilities
            line = {
                'lanes': lanes,
                'h_samples': self.scaled_h_samples,
                'raw_file': names[i]
            }
            line = json.dumps(line)
            self.out_file.write(line + '\n')


class TestOut:
    """
    This module allows to validate predictions against ground truth labels. It prints the accuracy after the test completed.
    Additionally it writes its results as csv to the directory where the trained model is located.
    """
    def __init__(self, config):
        # path to test.json
        self.out_dir = config.output_params.out_dir
        self.test_gt = config.output_params.test_gt
        self.trained_model = config.test_params.trained_model
        self.scaled_h_samples = config.data_params.scaled_h_samples
        self.lanes_pred = []
        self.LaneEval = LaneEval

    def out(self, predictions, names, _):
        """
        Collect results of batch

        Args:
            predictions: network result (list of samples containing probabilities per sample)
            names: filenames for predictions, if empty
        """
        if not names:
            raise Exception('Test output module requires "names". You probably either selected the wrong input or output module.')

        for i in range(len(predictions)):
            # get x coordinates based on probabilities
            lanes = evaluate_predictions(predictions[i])
            line = {
                'lanes': lanes,
                'h_samples': self.scaled_h_samples,
                'raw_file': names[i],
                'run_time': 10
            }
            self.lanes_pred.append(line)

    def post(self):
        """
        Evaluate collected data and print accuracy
        """
        res = self.LaneEval.bench_one_submit(self.lanes_pred, self.test_gt)
        res = json.loads(res)

        with open(os.path.join(self.out_dir, 'test_results.txt'), 'a') as f:
            print(f"Writing result file to {self.out_dir}")
            print(f"Model used: {self.trained_model}")
            f.write(f"Model used: {self.trained_model}\n")
            for r in res:
                print(f"{r['name']}: {r['value']}")
                f.write(f"{r['name']}: {r['value']}\n")
            f.write("\n")
        f.close()


class ProdOut:
    """
    You can use this class as a starting point to implement your "real use case"
    """

    def out(self, predictions, frames):
        """
        This is the place where you implement your out-logic.
        notice that you'll receive a list of predictions and filenames. The list list size equals your batch size.

        You will probably want to find the most probable class first
        or do more complex things to get more accurate class assignments

        Next step will be mapping the predicted class in a format suitable for your use case.
        Eg for image export this would be image coordinates

        Args:
            predictions: network result (list of samples containing probabilities per sample)
            names: filenames for predictions, might not be available depending on input module
            frames: source frames, might not be available depending on input module
        """
        for i in range(0, len(predictions)):
            print(predictions[i])

    def post(self):
        """
        Called after dataset/video/whatever was completely processed. You can do things like cleanup or printing stats here
        """
        print("finished")
