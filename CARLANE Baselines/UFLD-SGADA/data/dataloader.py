import os
import torch
import torchvision.transforms as transforms
import data.mytransforms as mytransforms
from data.constant import tusimple_row_anchor
from data.dataset import LaneDatasetTrainLabeled, LaneDatasetTrainUnlabeled, LaneDatasetTrainPseudo, LaneTestDataset


def get_loader_labeled(batch_size, data_root, file_name, griding_num, use_aux, distributed, num_lanes, num_workers):
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((36, 100)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200)
    ])

    train_dataset = LaneDatasetTrainLabeled(data_root,
                                            file_name,
                                            img_transform=img_transform,
                                            simu_transform=simu_transform,
                                            griding_num=griding_num,
                                            row_anchor=tusimple_row_anchor,
                                            segment_transform=segment_transform,
                                            use_aux=use_aux,
                                            num_lanes=num_lanes)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return train_loader


def get_loader_unlabeled(batch_size, data_root, file_name, distributed, num_workers):
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = LaneDatasetTrainUnlabeled(data_root,
                                              file_name,
                                              img_transform=img_transform)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return train_loader


def get_loader_pseudo_train(batch_size, data_root, file_name, griding_num, use_aux, distributed, num_lanes, num_workers):
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((36, 100)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200)
    ])

    train_dataset = LaneDatasetTrainPseudo(data_root,
                                           file_name,
                                           img_transform=img_transform,
                                           simu_transform=simu_transform,
                                           griding_num=griding_num,
                                           row_anchor=tusimple_row_anchor,
                                           segment_transform=segment_transform,
                                           use_aux=use_aux,
                                           num_lanes=num_lanes)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return train_loader


def get_loader_pseudo_label_generator(batch_size, data_root, file_name, num_workers):
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = LaneTestDataset(data_root,
                                    file_name,
                                    img_transform=img_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader


class SeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    '''
    Change the behavior of DistributedSampler to sequential distributed sampling.
    The sequential sampling helps the stability of multi-thread testing, which needs multi-thread file io.
    Without sequentially sampling, the file io on thread may interfere other threads.
    '''

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas, rank, shuffle)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        num_per_rank = int(self.total_size // self.num_replicas)

        # sequential sampling
        indices = indices[num_per_rank * self.rank: num_per_rank * (self.rank + 1)]

        assert len(indices) == self.num_samples

        return iter(indices)
