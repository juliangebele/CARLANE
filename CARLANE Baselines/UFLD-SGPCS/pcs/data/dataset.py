import torch
import os
import pdb
import numpy as np
from PIL import Image
from pcs.data.mytransforms import find_start_pos


class LaneDatasetTest(torch.utils.data.Dataset):
    def __init__(self, data_root, list_path, img_transform):
        super(LaneDatasetTest, self).__init__()
        self.data_root = data_root
        self.img_transform = img_transform

        with open(list_path, 'r') as f:
            self.list = [line for line in f.readlines() if line != '\n']

        # exclude the incorrect path prefix '/' of CULane
        self.list = [line[1:] if line[0] == '/' else line for line in self.list]

    def __getitem__(self, index):
        img_path = self.list[index].split()[0]
        img = Image.open(os.path.join(self.data_root, img_path))

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, img_path

    def __len__(self):
        return len(self.list)


class LaneDatasetTrainLabeled(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 list_path,
                 img_transform=None,
                 segment_transform=None,
                 simu_transform=None,
                 griding_num=100,
                 row_anchor=None,
                 use_aux=False,
                 num_lanes=2):
        super(LaneDatasetTrainLabeled, self).__init__()
        self.data_root = data_root
        self.img_transform = img_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform
        self.griding_num = griding_num
        self.use_aux = use_aux
        self.num_lanes = num_lanes

        with open(list_path, 'r') as f:
            self.list = f.readlines()

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        label_path = os.path.join(self.data_root, label_name)
        label = Image.open(label_path)

        img_path = os.path.join(self.data_root, img_name)
        image = Image.open(img_path)

        # Random Rotate or Offset Transformation
        if self.simu_transform is not None:
            image, label = self.simu_transform(image, label)

        # get the coordinates of lanes at row anchors
        lane_pts = self._get_index(label)

        w, h = image.size
        # make the coordinates to classification label
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)

        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        if self.img_transform is not None:
            image = self.img_transform(image)

        if self.use_aux:
            return index, image, cls_label, seg_label

        return index, image, cls_label

    def __len__(self):
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):
        w, h = label.size

        if h != 288:
            scale_f = lambda x: int((x * 1.0 / 288) * h)
            sample_tmp = list(map(scale_f, self.row_anchor))

        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))
        for i, r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, self.num_lanes + 1):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        # data augmentation: extend the lane to the boundary of image

        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i, :, 1] == -1):
                continue
            # if there is no lane

            valid = all_idx_cp[i, :, 1] != -1
            # get all valid lane points' index
            valid_idx = all_idx_cp[i, valid, :]
            # get all valid lane points
            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
                # this means this lane has reached the bottom boundary of the image
                # so we skip
                continue
            if len(valid_idx) < 6:
                continue
            # if the lane is too short to extend

            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)
            start_line = valid_idx_half[-1, 0]
            pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1

            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])
            fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])

            assert np.all(all_idx_cp[i, pos:, 1] == -1)
            all_idx_cp[i, pos:, 1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp


class LaneDatasetTrainUnlabeled(torch.utils.data.Dataset):
    def __init__(self, data_root, list_path, img_transform):
        super(LaneDatasetTrainUnlabeled, self).__init__()
        self.data_root = data_root
        self.img_transform = img_transform

        with open(list_path, 'r') as f:
            self.list = f.readlines()

    def __getitem__(self, index):
        img_path = self.list[index].split()[0]
        image = Image.open(os.path.join(self.data_root, img_path))

        if self.img_transform is not None:
            image = self.img_transform(image)

        return index, image

    def __len__(self):
        return len(self.list)


class LaneDatasetTrainPseudo(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 list_path,
                 img_transform=None,
                 segment_transform=None,
                 simu_transform=None,
                 griding_num=100,
                 row_anchor=None,
                 use_aux=False,
                 num_lanes=2):
        super(LaneDatasetTrainPseudo, self).__init__()
        self.data_root = data_root
        self.img_transform = img_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform
        self.griding_num = griding_num
        self.use_aux = use_aux
        self.num_lanes = num_lanes

        with open(list_path, 'r') as f:
            self.list = f.readlines()

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        cls_conf = l_info[-1]
        cls_conf = torch.tensor(float(cls_conf))
        label_path = os.path.join(self.data_root, label_name)
        label = Image.open(label_path)

        img_path = os.path.join(self.data_root, img_name)
        image = Image.open(img_path)

        # Random Rotate or Offset Transformation
        if self.simu_transform is not None:
            image, label = self.simu_transform(image, label)

        # get the coordinates of lanes at row anchors
        lane_pts = self._get_index(label)

        w, h = image.size
        # make the coordinates to classification label
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)

        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        if self.img_transform is not None:
            image = self.img_transform(image)

        if self.use_aux:
            return index, image, cls_label, seg_label, cls_conf

        return index, image, cls_label, cls_conf

    def __len__(self):
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):
        w, h = label.size

        if h != 288:
            scale_f = lambda x: int((x * 1.0 / 288) * h)
            sample_tmp = list(map(scale_f, self.row_anchor))

        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))
        for i, r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, self.num_lanes + 1):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        # data augmentation: extend the lane to the boundary of image

        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i, :, 1] == -1):
                continue
            # if there is no lane

            valid = all_idx_cp[i, :, 1] != -1
            # get all valid lane points' index
            valid_idx = all_idx_cp[i, valid, :]
            # get all valid lane points
            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
                # this means this lane has reached the bottom boundary of the image
                # so we skip
                continue
            if len(valid_idx) < 6:
                continue
            # if the lane is too short to extend

            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)
            start_line = valid_idx_half[-1, 0]
            pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1

            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])
            fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])

            assert np.all(all_idx_cp[i, pos:, 1] == -1)
            all_idx_cp[i, pos:, 1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp
