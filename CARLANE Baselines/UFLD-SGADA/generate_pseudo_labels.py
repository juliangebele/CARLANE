import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import warnings
import cv2
import tqdm
from scipy import special
from model.model import CNN, Discriminator
from utils.common import merge_config
from utils.dist_utils import dist_print, dist_tqdm
from data.dataloader import get_loader_pseudo_label_generator


@torch.no_grad()
def generate_pseudo_labels(net, disc, data_loader):
    net.eval()
    disc.eval()

    image_height = 720
    row_anchor_start = 160
    h_samples = [x/image_height for x in range(row_anchor_start, image_height, 10)]
    scaled_h_samples = [int(round(x * image_height)) for x in h_samples]

    data_iter = iter(data_loader)

    progress_bar = dist_tqdm(data_loader, desc=f"[Compute Pseudo Labels]")
    for _ in progress_bar:
        image, path = next(data_iter)
        image = image.cuda()

        feature = net.encoder(image)
        cls_out = net.classifier(feature)
        disc_out = disc(feature)

        # evaluate preditions
        for i in range(cls_out.size(0)):
            # Discriminator probabilities and pseudo-labels
            d_prob, d_label = torch.topk(disc_out[i], k=1, dim=0)
            d_prob, d_label = torch.squeeze(d_prob), torch.squeeze(d_label)

            # Classifier probabilities and pseudo-labels
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

            mean_prob = (1.0 * mean_prob_lane + 1.0 * mean_prob_nolane) / 2.0
            # print(f"mean_prob_lane: {mean_prob_lane}, mean_prob_nolane: {mean_prob_nolane}")

            # Process on CPU with numpy
            # output = cls_out[i].data.cpu().numpy()
            # cls_label = np.argmax(output, axis=0)  # get most probable x-class per lane and h_sample (56,4)
            # cls_prob = special.softmax(output[:-1, :, :], axis=0)  # (100, 56, 4)
            # idx = np.arange(100)
            # idx = idx.reshape(-1, 1, 1)
            # loc = np.sum(cls_prob * idx, axis=0)    # (56, 4)
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

            # write confidences (cls_conf, disc_prediction, disc_confidence)
            confidences = path[i] + ' ' + str(mean_prob.item()) + ' ' + str(d_label.item()) + ' ' + str(d_prob.item())
            confidence_file.write(confidences + '\n')


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

    for i in tqdm.tqdm(range(len(line_txt))):
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

    for i in tqdm.tqdm(range(len(line_txt))):
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


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # load data
    train_loader_target = get_loader_pseudo_label_generator(cfg.batch_size, cfg.data_root, cfg.target_train, cfg.num_workers)

    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    cls_dim = (cfg.griding_num+1, cfg.cls_num_per_lane, cfg.num_lanes)
    target_cnn = CNN(backbone=cfg.backbone, cls_dim=cls_dim, use_aux=False).cuda()  # we dont need auxiliary segmentation in testing
    discriminator = Discriminator(slope=cfg.slope).cuda()

    dist_print('Loading models...')

    # Load model
    adda_state_dict = torch.load(cfg.pretrained)
    adda_state_dict = {(key.replace("module.", "") if "module." in key else key): value for key, value in adda_state_dict.items()}

    # Encoder
    encoder_state_dict = {key: value for key, value in adda_state_dict["encoder"].items() if "model" in key or "aux" in key or "pool" in key}
    target_cnn.encoder.load_state_dict(encoder_state_dict, strict=False)

    # Classifier
    cls_state_dict = {key: value for key, value in adda_state_dict["classifier"].items() if "cls" in key}
    target_cnn.classifier.load_state_dict(cls_state_dict, strict=False)

    # Discriminator
    disc_state_dict = {key: value for key, value in adda_state_dict["discriminator"].items() if "discriminator" in key}
    discriminator.load_state_dict(disc_state_dict, strict=False)

    dist_print(f"Model successfully loaded from '{cfg.pretrained}'\n")

    data_root = cfg.data_root
    pseudo_label_file = open(os.path.join(data_root, 'pseudo_labels.json'), 'w')
    confidence_file = open(os.path.join(data_root, 'confidence_file.txt'), 'w')

    generate_pseudo_labels(target_cnn, discriminator, train_loader_target)

    pseudo_label_file.close()
    confidence_file.close()

    # convert .json to .png label images
    train_files = "pseudo_labels.json"
    new_file = "pseudo_labels.txt"

    print("Generating .png labels...")

    names, line_txt = get_tusimple_list(data_root, train_files.split(','))

    if cfg.num_lanes == 2:
        generate_segmentation_and_train_list_2lanes(data_root, new_file, line_txt, names)
    elif cfg.num_lanes == 4:
        generate_segmentation_and_train_list_4lanes(data_root, new_file, line_txt, names)

    print("Generating .txt file...")

    # merge pseudo labels and confidence file
    with open(os.path.join(data_root, 'pseudo_labels.txt'), 'r') as f:
        list_a = f.readlines()

    with open(os.path.join(data_root, 'confidence_file.txt'), 'r') as f:
        list_b = f.readlines()

    list_c = open(cfg.target_train_pseudo, "w")
    print(f'Placing pseudo label file in {cfg.target_train_pseudo}')

    search_paths = []
    for line in list_b:
        search_paths.append(line.split()[0])

    for i in range(len(list_a)):
        pseudo_label_list = list_a[i].split()
        index = search_paths.index(pseudo_label_list[0])
        confidence_list = list_b[index].split()

        line = ''
        for j in range(len(pseudo_label_list)):
            line += pseudo_label_list[j] + ' '
        line += confidence_list[1] + ' ' + confidence_list[2] + ' ' + confidence_list[3]
        list_c.write(line + '\n')
    list_c.close()

    os.remove(os.path.join(data_root, "pseudo_labels.json"))
    os.remove(os.path.join(data_root, "pseudo_labels.txt"))
    os.remove(os.path.join(data_root, "confidence_file.txt"))
