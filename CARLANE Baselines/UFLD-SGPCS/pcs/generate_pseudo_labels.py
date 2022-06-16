import os
import json
import warnings
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from dotmap import DotMap
from scipy import special
from pcs.utils import load_json, set_default
from pcs.data.datautils import create_dataset_test, create_loader
from pcs.models.model import Net
from pcs.models.head import Classifier, CosineClassifier


class PseudoLabelAgent(object):
    def __init__(self, config):
        self.config = config

    def load_data(self):
        data_root = self.config.data_params.data_root

        images_input_file = self.config.input_params.images_input_file
        batch_size = self.config.model_params.batch_size
        num_workers = self.config.data_params.num_workers

        image_height_net = self.config.data_params.image_height_net
        image_width_net = self.config.data_params.image_width_net

        dataset_pseudo_labels = create_dataset_test(data_root, images_input_file, image_height=image_height_net, image_width=image_width_net)
        loader = create_loader(dataset_pseudo_labels, batch_size, num_workers, is_train=False)

        return loader

    def load_model(self):
        """
        Setup and load neural network
        """
        trained_model = self.config.test_params.trained_model
        backbone = self.config.model_params.backbone.split("-")[1]
        griding_num = self.config.model_params.griding_num
        h_samples = self.config.data_params.h_samples
        num_lanes = self.config.model_params.num_lanes
        cls_dim = (griding_num + 1, len(h_samples), num_lanes)
        temp = self.config.model_params.cls_temp

        assert backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

        torch.backends.cudnn.benchmark = True
        model = Net(pretrained=False, backbone=backbone, cls_dim=cls_dim, use_aux=False).cuda()

        # Load PCS_UFLD model
        if trained_model.endswith(".tar"):
            cls_head = CosineClassifier(cls_dim=cls_dim, temp=temp).cuda()
            trained_model_state_dict = torch.load(trained_model, map_location="cpu")

            model_state_dict = {(key.replace("module.", "") if "module." in key else key): value for key, value in trained_model_state_dict["model_state_dict"].items()}
            model.load_state_dict(model_state_dict, strict=False)

            if "cls_state_dict" in trained_model_state_dict:
                cls_state_dict = trained_model_state_dict["cls_state_dict"]
                cls_head.load_state_dict(cls_state_dict)

            print(f"Model successfully loaded from '{trained_model}', val acc = {trained_model_state_dict['val_acc']}\n")

        # Load UFLD model
        elif trained_model.endswith(".pth"):
            cls_head = Classifier(cls_dim=cls_dim).cuda()
            if 'DANN' in trained_model or 'ADDA' in trained_model or 'SGADA' in trained_model:
                trained_model_state_dict = torch.load(trained_model, map_location="cpu")["encoder"]
            else:
                trained_model_state_dict = torch.load(trained_model, map_location="cpu")["model"]
            trained_model_state_dict = {(key.replace("module.", "") if "module." in key else key): value for key, value in trained_model_state_dict.items()}

            model_state_dict = {key: value for key, value in trained_model_state_dict.items() if "model" in key or "aux" in key or "pool" in key}
            model.load_state_dict(model_state_dict, strict=False)

            if 'DANN' in trained_model or 'ADDA' in trained_model or 'SGADA' in trained_model:
                trained_model_state_dict = torch.load(trained_model, map_location="cpu")["classifier"]
                trained_model_state_dict = {(key.replace("module.", "") if "module." in key else key): value for key, value in trained_model_state_dict.items()}
            cls_state_dict = {key: value for key, value in trained_model_state_dict.items() if "cls" in key}
            cls_head.load_state_dict(cls_state_dict, strict=False)

            print(f"Model successfully loaded from '{trained_model}'\n")
        else:
            raise NotImplementedError("Model not supported or not properly loaded")

        model.eval()
        cls_head.eval()

        return model, cls_head

    @torch.no_grad()
    def get_pseudo_labels(self, enc, cls, data_loader):
        enc.eval()
        cls.eval()

        data_root = self.config.data_params.data_root
        scaled_h_samples = self.config.data_params.scaled_h_samples

        pseudo_label_file = open(os.path.join(data_root, 'pseudo_labels.json'), 'w')
        confidence_file = open(os.path.join(data_root, 'confidence_file.txt'), 'w')

        data_iter = iter(data_loader)

        progress_bar = tqdm(data_loader, desc=f"[Compute Pseudo Labels]")
        for _ in progress_bar:
            image, path = next(data_iter)
            image = image.cuda()

            feature = enc(image)
            cls_out = cls(feature)

            # evaluate preditions for each image
            for i in range(cls_out.size(0)):
                soft_out = F.softmax(cls_out[i], dim=0)  # [101, 56, 2]
                soft_out = soft_out.permute((2, 1, 0))  # [2, 56, 101]

                prob, ind = torch.topk(soft_out, k=1, dim=2)  # [2, 56, 1]
                prob = torch.squeeze(prob)  # [2, 56]
                ind = torch.squeeze(ind)

                mean_prob_lane = []
                mean_prob_nolane = []
                for lane_i in range(soft_out.size(0)):
                    # get all valid indices (lane predictions)
                    valid_lane = ind[lane_i] < 100
                    valid_mask_lane = valid_lane.nonzero(as_tuple=False)[:, 0]
                    prob_lane = prob[lane_i, valid_mask_lane]
                    mean_prob_lane.append(prob_lane)

                    # get all valid indices for absent lanes (no lane predictions)
                    valid_nolane = ind[lane_i] == 100
                    valid_mask_nolane = valid_nolane.nonzero(as_tuple=False)[:, 0]
                    prob_nolane = prob[lane_i, valid_mask_nolane]
                    mean_prob_nolane.append(prob_nolane)

                # merge probabilities of both lanes into one tensor
                mean_prob_lane = torch.cat(mean_prob_lane, dim=0)
                mean_prob_nolane = torch.cat(mean_prob_nolane, dim=0)

                # calculate mean for both cases (present and absent lane preds)
                if mean_prob_lane.numel() == 0:
                    mean_prob_lane = torch.tensor(0)
                else:
                    mean_prob_lane = mean_prob_lane.mean()
                if mean_prob_nolane.numel() == 0:
                    mean_prob_nolane = torch.tensor(0)
                else:
                    mean_prob_nolane = mean_prob_nolane.mean()

                mean_prob = (mean_prob_lane + mean_prob_nolane) / 2.0
                # print(f"mean_prob_lane: {mean_prob_lane}, mean_prob_nolane: {mean_prob_nolane}")

                # Process on CPU with numpy
                # output = cls_out[i].data.cpu().numpy()
                # cls_label = np.argmax(output, axis=0)  # get most probable x-class per lane and h_sample (56,4)
                # cls_prob = special.softmax(output[:-1, :, :], axis=0)  # (100, 56, 4)
                # idx = np.arange(100)
                # idx = idx.reshape(-1, 1, 1)
                # loc = np.sum(cls_prob * idx, axis=0)  # (56, 4)
                # loc[cls_label == 100] = -2
                # cls_label = loc

                # Process on GPU with pytorch
                output = cls_out[i]
                cls_label = torch.argmax(output, dim=0)  # get most probable x-class per lane and h_sample
                cls_prob = F.softmax(output[:-1, :, :], dim=0)
                idx = torch.arange(100).cuda()
                idx = idx.reshape(-1, 1, 1)
                loc = torch.sum(cls_prob * idx, dim=0)
                loc[cls_label == 100] = -2
                cls_label = loc

                # Map x-axis (griding_num) estimations to image coordinates
                lanes = []
                offset = 0.5  # different values used in ufld project. demo: 0.0, test: 0.5
                for j in range(cls_label.shape[1]):
                    out_j = cls_label[:, j]
                    lane = [int(torch.round((loc + offset) * 1280.0 / (100 - 1))) if loc != -2 else -2 for loc in out_j]  # on GPU
                    # lane = [int(round((loc + offset) * 1280.0 / (100 - 1))) if loc != -2 else -2 for loc in out_j]  # on CPU
                    # lane = [int((loc + offset) * float(1280) / (100 - 1)) if loc != -2 else -2 for loc in out_j]
                    lanes.append(lane)

                line = {
                    'lanes': lanes,
                    'h_samples': scaled_h_samples,
                    'raw_file': path[i]
                }
                line = json.dumps(line)
                pseudo_label_file.write(line + '\n')

                # write confidences to file
                confidences = path[i] + ' ' + str(mean_prob.item())
                confidence_file.write(confidences + '\n')

        pseudo_label_file.close()
        confidence_file.close()


def calc_k(line):
    """
    Calculate the direction of lanes
    """
    line_x = line[::2]
    line_y = line[1::2]
    length = np.sqrt((line_x[0] - line_x[-1]) ** 2 + (line_y[0] - line_y[-1]) ** 2)
    if length < 1:     # changed from 90 to 1
        return -10  # if the lane is too short, it will be skipped

    p = np.polyfit(line_x, line_y, deg=1)
    rad = np.arctan(p[0])

    return rad


def draw(im, line, idx, show=False):
    """
    Generate the segmentation label according to json annotation
    """
    line_x = line[::2]
    line_y = line[1::2]
    pt0 = (int(line_x[0]), int(line_y[0]))
    if show:
        cv2.putText(im, str(idx), (int(line_x[len(line_x) // 2]), int(line_y[len(line_x) // 2]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        idx = idx * 60

    for i in range(len(line_x) - 1):
        cv2.line(im, pt0, (int(line_x[i + 1]), int(line_y[i + 1])), (idx,), thickness=16)
        pt0 = (int(line_x[i + 1]), int(line_y[i + 1]))


def get_tusimple_list(root, label_list):
    """
    Get all the files' names from the json annotation
    """
    label_json_all = []
    for l in label_list:
        l = os.path.join(root, l)
        label_json = [json.loads(line) for line in open(l).readlines()]
        label_json_all += label_json
    names = [l['raw_file'] for l in label_json_all]
    h_samples = [np.array(l['h_samples']) for l in label_json_all]
    lanes = [np.array(l['lanes']) for l in label_json_all]

    line_txt = []
    for i in range(len(lanes)):
        line_txt_i = []
        for j in range(len(lanes[i])):
            if np.all(lanes[i][j] == -2):
                continue
            valid = lanes[i][j] != -2
            line_txt_tmp = [None] * (len(h_samples[i][valid]) + len(lanes[i][j][valid]))
            line_txt_tmp[::2] = list(map(str, lanes[i][j][valid]))
            line_txt_tmp[1::2] = list(map(str, h_samples[i][valid]))
            line_txt_i.append(line_txt_tmp)
        line_txt.append(line_txt_i)

    return names, line_txt


def generate_segmentation_and_train_list_2lanes(root, new_file, line_txt, names):
    """
    The lane annotations of the Tusimple dataset is not strictly in order, so we need to find out the correct lane order for segmentation.
    We use the same definition as CULane, in which the four lanes from left to right are represented as 1,2,3,4 in segentation label respectively.
    """
    train_gt_fp = open(os.path.join(root, new_file), 'w')

    for i in tqdm(range(len(line_txt))):
        tmp_line = line_txt[i]
        if len(tmp_line) == 0:
            # quick fix for invalid input data
            print('no lines on current sample', flush=True)
            continue
        lines = []
        for j in range(len(tmp_line)):
            lines.append(list(map(float, tmp_line[j])))

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                ks = np.array([calc_k(line) for line in lines])  # get the direction of each lane
            except np.RankWarning:
                print('rank warning', flush=True)
                continue

        k_neg = ks[ks < 0].copy()
        k_pos = ks[ks > 0].copy()
        k_neg = k_neg[k_neg != -10]  # -10 means the lane is too short and is discarded
        k_pos = k_pos[k_pos != -10]
        k_neg.sort()
        k_pos.sort()

        label_path = names[i][:-3] + 'png'
        label = np.zeros((720, 1280), dtype=np.uint8)
        bin_label = [0, 0]

        if len(k_neg) == 1:  # for only one lane in the left
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label, lines[which_lane], 1)
            bin_label[0] = 1
        elif len(k_neg) == 2:  # some edge cases which have 2 left lanes and 0 right lane
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label, lines[which_lane], 1)
            bin_label[0] = 1

            which_lane = np.where(ks == k_neg[1])[0][0]
            draw(label, lines[which_lane], 2)
            bin_label[1] = 1

        if len(k_pos) == 1:  # For the lanes in the right, the same logical is adopted.
            which_lane = np.where(ks == k_pos[0])[0][0]
            draw(label, lines[which_lane], 2)
            bin_label[1] = 1
        elif len(k_pos) == 2:  # some edge cases which have 2 left lanes and 0 right lane
            which_lane = np.where(ks == k_pos[0])[0][0]
            draw(label, lines[which_lane], 1)
            bin_label[0] = 1

            which_lane = np.where(ks == k_pos[1])[0][0]
            draw(label, lines[which_lane], 2)
            bin_label[1] = 1

        cv2.imwrite(os.path.join(root, label_path), label)

        train_gt_fp.write(names[i] + ' ' + label_path + ' ' + ' '.join(list(map(str, bin_label))) + '\n')
    train_gt_fp.close()


def generate_segmentation_and_train_list_4lanes(root, new_file, line_txt, names):
    """
    The lane annotations of the Tusimple dataset is not strictly in order, so we need to find out the correct lane order for segmentation.
    We use the same definition as CULane, in which the four lanes from left to right are represented as 1,2,3,4 in segentation label respectively.
    """
    train_gt_fp = open(os.path.join(root, new_file), 'w')

    for i in tqdm(range(len(line_txt))):
        tmp_line = line_txt[i]
        if len(tmp_line) == 0:
            # quick fix for invalid input data
            print('no lines on current sample', flush=True)
            continue
        lines = []
        for j in range(len(tmp_line)):
            lines.append(list(map(float, tmp_line[j])))

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                ks = np.array([calc_k(line) for line in lines])  # get the direction of each lane
            except np.RankWarning:
                print('rank warning', flush=True)
                continue

        k_neg = ks[ks < 0].copy()
        k_pos = ks[ks > 0].copy()
        k_neg = k_neg[k_neg != -10]  # -10 means the lane is too short and is discarded
        k_pos = k_pos[k_pos != -10]
        k_neg.sort()
        k_pos.sort()

        label_path = names[i][:-3] + 'png'
        label = np.zeros((720, 1280), dtype=np.uint8)
        bin_label = [0, 0, 0, 0]

        if len(k_neg) == 1:  # for only one lane in the left
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label, lines[which_lane], 2)
            bin_label[1] = 1
        elif len(k_neg) == 2:  # for two lanes in the left
            which_lane = np.where(ks == k_neg[1])[0][0]
            draw(label, lines[which_lane], 1)
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label, lines[which_lane], 2)
            bin_label[0] = 1
            bin_label[1] = 1
        elif len(k_neg) > 2:  # for more than two lanes in the left,
            which_lane = np.where(ks == k_neg[1])[0][0]  # we only choose the two lanes that are closest to the center
            draw(label, lines[which_lane], 1)
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label, lines[which_lane], 2)
            bin_label[0] = 1
            bin_label[1] = 1

        if len(k_pos) == 1:  # For the lanes in the right, the same logical is adopted.
            which_lane = np.where(ks == k_pos[0])[0][0]
            draw(label, lines[which_lane], 3)
            bin_label[2] = 1
        elif len(k_pos) == 2:
            which_lane = np.where(ks == k_pos[1])[0][0]
            draw(label, lines[which_lane], 3)
            which_lane = np.where(ks == k_pos[0])[0][0]
            draw(label, lines[which_lane], 4)
            bin_label[2] = 1
            bin_label[3] = 1
        elif len(k_pos) > 2:
            which_lane = np.where(ks == k_pos[-1])[0][0]
            draw(label, lines[which_lane], 3)
            which_lane = np.where(ks == k_pos[-2])[0][0]
            draw(label, lines[which_lane], 4)
            bin_label[2] = 1
            bin_label[3] = 1

        cv2.imwrite(os.path.join(root, label_path), label)

        train_gt_fp.write(names[i] + ' ' + label_path + ' ' + ' '.join(list(map(str, bin_label))) + '\n')
    train_gt_fp.close()


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
    parser.add_argument("--config", type=str, default="config/tulane_test.json", help="the path to the config")

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


def generate_pseudo_labels(config):
    pseudo_agent = PseudoLabelAgent(config)

    test_loader = pseudo_agent.load_data()
    encoder, classifier = pseudo_agent.load_model()

    pseudo_agent.get_pseudo_labels(encoder, classifier, test_loader)

    # convert .json to .png label images
    data_path = config.data_params.data_root
    train_files = "pseudo_labels.json"
    new_file = "pseudo_labels.txt"

    print("Generating .png labels...")

    names, line_txt = get_tusimple_list(data_path, train_files.split(","))

    if config.model_params.num_lanes == 2:
        generate_segmentation_and_train_list_2lanes(data_path, new_file, line_txt, names)
    elif config.model_params.num_lanes == 4:
        generate_segmentation_and_train_list_4lanes(data_path, new_file, line_txt, names)

    print("Generating .txt file...")

    # merge pseudo labels and confidence file
    with open(os.path.join(data_path, "pseudo_labels.txt"), "r") as f:
        list_a = f.readlines()

    with open(os.path.join(data_path, "confidence_file.txt"), "r") as f:
        list_b = f.readlines()

    pseudo_label_file = os.path.split(config.input_params.images_input_file)[0]
    list_c = open(os.path.join(pseudo_label_file, "target_train_pseudo.txt"), "w")
    print(f'Placing pseudo label file in {os.path.join(pseudo_label_file, "target_train_pseudo.txt")}')

    search_paths = []
    for line in list_b:
        search_paths.append(line.split()[0])

    for i in range(len(list_a)):
        pseudo_label_list = list_a[i].split()
        index = search_paths.index(pseudo_label_list[0])
        confidence_list = list_b[index].split()

        line = ""
        for j in range(len(pseudo_label_list)):
            line += pseudo_label_list[j] + " "
        line += confidence_list[1]
        list_c.write(line + "\n")
    list_c.close()

    os.remove(os.path.join(data_path, "pseudo_labels.json"))
    os.remove(os.path.join(data_path, "pseudo_labels.txt"))
    os.remove(os.path.join(data_path, "confidence_file.txt"))


if __name__ == "__main__":
    init_parser = init_parser()
    args = init_parser.parse_args()

    # load config
    cfg = load_json(args.config)

    # json to DotMap
    cfg = DotMap(cfg)
    cfg = adjust_config(cfg)

    generate_pseudo_labels(cfg)
