import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pcs.data.dataset import LaneDatasetTrainLabeled, LaneDatasetTrainUnlabeled, LaneDatasetTrainPseudo, LaneDatasetTest
import pcs.data.mytransforms as mytransforms


# Fewshot Unsupervised Domain Adaptation is not supported
def get_fewshot_index(lbd_dataset, whl_dataset):
    lbd_imgs = lbd_dataset.imgs
    whl_imgs = whl_dataset.imgs
    fewshot_indices = [whl_imgs.index(path) for path in lbd_imgs]
    fewshot_labels = lbd_dataset.labels
    return fewshot_indices, fewshot_labels


# Dataset


def create_dataset_lbd(data_root, list_path, image_height=288, image_width=800, griding_num=100, row_anchor=None, num_lanes=2, use_aux=False, data_aug="no_aug"):
    segment_transform = transforms.Compose([mytransforms.FreeScaleMask((image_height // 8, image_width // 8)),
                                            mytransforms.MaskToTensor()])

    simu_transform = mytransforms.Compose2([mytransforms.RandomRotate(6),
                                            mytransforms.RandomUDoffsetLABEL(100),
                                            mytransforms.RandomLROffsetLABEL(200)])

    img_transforms = get_augmentation(image_height, image_width)

    return LaneDatasetTrainLabeled(data_root,
                                   list_path,
                                   img_transform=img_transforms[data_aug],
                                   segment_transform=segment_transform,
                                   simu_transform=simu_transform,
                                   griding_num=griding_num,
                                   row_anchor=row_anchor,
                                   num_lanes=num_lanes,
                                   use_aux=use_aux)


def create_dataset_unl(data_root, list_path, image_height=288, image_width=800, data_aug="no_aug"):

    img_transforms = get_augmentation(image_height, image_width)

    return LaneDatasetTrainUnlabeled(data_root,
                                     list_path,
                                     img_transform=img_transforms[data_aug])


def create_dataset_pseudo(data_root, list_path, image_height=288, image_width=800, griding_num=100, row_anchor=None, num_lanes=2, use_aux=False, data_aug="no_aug"):
    segment_transform = transforms.Compose([mytransforms.FreeScaleMask((image_height // 8, image_width // 8)),
                                            mytransforms.MaskToTensor()])

    simu_transform = mytransforms.Compose2([mytransforms.RandomRotate(6),
                                            mytransforms.RandomUDoffsetLABEL(100),
                                            mytransforms.RandomLROffsetLABEL(200)])

    img_transforms = get_augmentation(image_height, image_width)

    return LaneDatasetTrainPseudo(data_root,
                                  list_path,
                                  img_transform=img_transforms[data_aug],
                                  segment_transform=segment_transform,
                                  simu_transform=simu_transform,
                                  griding_num=griding_num,
                                  row_anchor=row_anchor,
                                  num_lanes=num_lanes,
                                  use_aux=use_aux)


def create_dataset_test(data_root, list_path, image_height=288, image_width=800):
    img_transform = transforms.Compose([transforms.Resize((image_height, image_width)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    return LaneDatasetTest(data_root,
                           list_path,
                           img_transform=img_transform)


# Dataloader

def get_augmentation(image_height, image_width):
    # PCS-Paper used a margin of 32 for 224x224 imgs
    # The ratio is (41/288, 114/800) which is equal to 32/224
    margin_height = 41
    margin_width = 114
    img_transforms = {
        # original pcs augs
        "raw": transforms.Compose([transforms.Resize((image_height + margin_height, image_width + margin_width)),
                                   transforms.CenterCrop((image_height, image_width)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "aug_0": transforms.Compose([transforms.Resize((image_height + margin_height, image_width + margin_width)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomCrop((image_height, image_width)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "aug_1": transforms.Compose([transforms.RandomResizedCrop((image_height, image_width), scale=(0.2, 1.0)),
                                     transforms.RandomGrayscale(p=0.2),
                                     transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),

        # custom augs
        "no_aug": transforms.Compose([transforms.Resize((image_height, image_width)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "soft_aug": transforms.Compose([transforms.Resize((image_height, image_width)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "pixel_aug": transforms.Compose([transforms.Resize((image_height, image_width)),
                                         transforms.RandomGrayscale(p=0.2),
                                         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "weak_aug": transforms.Compose([transforms.Resize((image_height, image_width)),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "heavy_aug": transforms.Compose([transforms.Resize((image_height, image_width)),
                                         transforms.RandomGrayscale(p=0.2),
                                         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                         transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 8.0)),
                                         # transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                         transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value='random', inplace=False)])
    }

    return img_transforms


def worker_init_seed(worker_id):
    np.random.seed(12 + worker_id)
    random.seed(12 + worker_id)


def create_loader(dataset, batch_size=32, num_workers=4, is_train=True):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=min(batch_size, len(dataset)),
                                       num_workers=num_workers,
                                       shuffle=is_train,
                                       drop_last=is_train,
                                       pin_memory=False,
                                       worker_init_fn=worker_init_seed)
