#!/usr/bin/env python3

import argparse
from dotmap import DotMap
from pcs.agents.TestAgent import TestAgent
from pcs.utils.utils import load_json, set_default


def adjust_config(config):
    set_default(config, "debug", value=False)
    set_default(config, "cuda", value=True)
    set_default(config, "gpu_device", value=None)

    # data_params
    set_default(config.data_params, "num_workers", value=2)
    set_default(config.data_params, "data_root", value=None)
    set_default(config.data_params, "image_height", value=720)
    set_default(config.data_params, "image_width", value=1280)
    set_default(config.data_params, "image_height_net", value=288)
    set_default(config.data_params, "image_width_net", value=800)
    set_default(config.data_params, "row_anchor_start", value=160)
    image_height = config.data_params.image_height
    row_anchor_start = config.data_params.row_anchor_start
    h_samples = [x/image_height for x in range(row_anchor_start, image_height, 10)]
    scaled_h_samples = [int(round(x * image_height)) for x in h_samples]
    set_default(config.data_params, "h_samples", value=h_samples)
    set_default(config.data_params, "scaled_h_samples", value=scaled_h_samples)

    # model_params
    set_default(config.model_params, "griding_num", value=100)
    set_default(config.model_params, "num_lanes", value=2)
    set_default(config.model_params, "use_aux", value=True)
    set_default(config.model_params, "cls_temp", value=0.05)
    set_default(config.model_params, "batch_size", value=8)

    # test_params
    set_default(config.test_params, "trained_model", value=None)
    set_default(config.test_params, "measure_time", value=False)
    set_default(config.test_params, "test_txt", value=None)

    # input_params
    set_default(config.input_params, "input_mode", value="images")
    set_default(config.input_params, "video_input_file", value=None)
    set_default(config.input_params, "camera_input_cam_number", value=[0, 0, 1920, 1080])
    set_default(config.input_params, "screencap_recording_area", value=None)
    set_default(config.input_params, "screencap_enable_image_forwarding", value=False)

    # output_params
    set_default(config.output_params, "output_mode", value=None)  # ["video", "test", "json"]
    set_default(config.output_params, "out_dir", value=None)
    set_default(config.output_params, "test_gt", value=None)
    set_default(config.output_params, "enable_live_video", value=True)
    set_default(config.output_params, "enable_video_export", value=True)
    set_default(config.output_params, "enable_image_export", value=False)
    set_default(config.output_params, "enable_line_mode", value=True)

    return config


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/molane_test.json", help="the path to the config")

    # Data
    parser.add_argument("--data_root", type=str, default=None, help="absolute path to root directory of your dataset")
    parser.add_argument("--img_height", type=int, default=None, help="height of input images")
    parser.add_argument("--img_width", type=int, default=None, help="width of input images")

    # Model
    parser.add_argument("--backbone", type=str, default=None, help="ResNet: ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']")
    parser.add_argument("--batch_size", type=int, default=None, help="number of samples to process in one batch")

    # Test params
    parser.add_argument("--trained_model", type=str, default=None, help="load trained model and use it for runtime")
    parser.add_argument("--measure_time", action="store_true", default=None, help="enable speed measurement")
    parser.add_argument("--test_txt", type=str, default=None, help="testing index file (test.txt)")

    # Input
    parser.add_argument("--input_mode", type=str, help="specifies input module")
    parser.add_argument("--video_input_file", type=str, default=None, help="full filepath to video file you want to use as input")
    parser.add_argument("--camera_input_cam_number", type=int, default=None, help="number of your camera")
    parser.add_argument("--screencap_recording_area", type=int, nargs="+", default=None, help="position and size of recording area: x,y,w,h")
    parser.add_argument("--screencap_enable_image_forwarding", action="store_false", default=None, help="allows disabling image forwarding. While this will probably improve performance for this input it will prevent you from using most out_modules as also no input_file (with paths to frames on disk) is available in this module")

    # Output
    parser.add_argument("--output_mode", type=str, action="append", help="specifies output module, can define multiple modules by using this parameter multiple times. Using multiple out-modules might decrease performance significantly")
    parser.add_argument("--out_dir", type=str, default=None, help="working directory: every output will be written here")
    parser.add_argument("--test_gt", type=str, default=None, help="file containing labels for test data to validate test results")
    parser.add_argument("--enable_live_video", action="store_false", default=None, help="enable/disable live preview")
    parser.add_argument("--enable_video_export", action="store_false", default=None, help="enable/disable export to video file")
    parser.add_argument("--enable_image_export", action="store_false", default=None, help="enable/disable export to images (like video, but as jpegs)")
    parser.add_argument("--enable_line_mode", action="store_false", default=None, help="enable/disable visualize as lines instead of points")

    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    # load config
    config_json = load_json(args.config)

    # json to DotMap
    config = DotMap(config_json)
    config = adjust_config(config)

    # create agent
    agent = TestAgent(config)

    try:
        agent.run()
    except KeyboardInterrupt:
        pass
