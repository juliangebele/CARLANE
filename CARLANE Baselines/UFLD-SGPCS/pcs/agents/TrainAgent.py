import os
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pcs.models import (Classifier, CosineClassifier, MemoryBank, SSDALossModule, loss_info, torch_kmeans, update_data_memory)
from pcs.models import (SoftmaxFocalLoss, ParsingRelationLoss, ParsingRelationDis)
from pcs.models.model import Net
from pcs.utils import (AverageMeter, is_div, torchutils, size_of_tensor)
from pcs.utils.metrics import update_metrics, reset_metrics, get_metric_dict
from pcs.data import datautils
from . import BaseAgent

ls_abbr = {
    "cls-so": "cls",
    "proto-each": "P",
    "proto-src": "Ps",
    "proto-tgt": "Pt",
    "cls-info": "info",
    "I2C-cross": "C",
    "semi-condentmax": "sCE",
    "semi-entmin": "sE",
    "tgt-condentmax": "tCE",
    "tgt-entmin": "tE",
    "semi-pseudo": "psd-src",
    "tgt-pseudo": "psd-tgt",
    "ID-each": "I",
    "CD-cross": "CD",
    "sim-src": "simS",
    "sim-tgt": "simT",
    "shape-src": "shpS",
    "shape-tgt": "shpT",
    "aux-src": "auxS",
    "aux-tgt": "auxT"
}


class TrainAgent(BaseAgent):
    def __init__(self, config):
        self.config = config
        self._define_task(config)
        self.is_features_computed = False
        self.current_iteration_source = self.current_iteration_target = 0

        self.griding_num = self.config.model_params.griding_num
        self.image_height = self.config.data_params.image_height
        self.row_anchor_start = self.config.model_params.row_anchor_start
        self.h_samples = [x / self.image_height for x in range(self.row_anchor_start, self.image_height, 10)]
        self.num_lanes = self.config.model_params.num_lanes
        self.cls_dim = (self.griding_num + 1, len(self.h_samples), self.num_lanes)
        self.num_class = np.prod(self.cls_dim)
        self.use_aux = self.config.model_params.use_aux
        self.use_l2_norm = self.config.model_params.use_l2_norm

        self.metric_dict = get_metric_dict(self.griding_num, self.num_lanes)
        wandb.init(project="UFLD-SGPCS", name="UFLD-SGPCS", config=self.config, dir=config.exp_base)

        self.domain_map = {
            "source": self.config.data_params.source,
            "target": self.config.data_params.target
        }

        super(TrainAgent, self).__init__(config)

        wandb.watch((self.model, self.cls_head), log="all")

        # for MIM
        self.momentum_softmax_source = torchutils.MomentumSoftmax(self.num_class, m=len(self.get_attr("source", "train_loader")))
        self.momentum_softmax_target = torchutils.MomentumSoftmax(self.num_class, m=len(self.get_attr("target", "train_loader")))

        # init loss
        loss_fn = SSDALossModule(self.config, gpu_devices=self.gpu_devices)
        self.loss_fn = nn.DataParallel(loss_fn, device_ids=self.gpu_devices).cuda()
        self.loss_cls = SoftmaxFocalLoss().cuda()
        self.loss_pseudo = SoftmaxFocalLoss().cuda()
        self.loss_sim = ParsingRelationLoss().cuda()
        self.loss_shape = ParsingRelationDis().cuda()
        self.loss_aux = nn.CrossEntropyLoss().cuda()

        if self.config.pretrained_exp_dir is None and self.config.pretrained_pcs_ufld is None and self.ssl:
            self._init_memory_bank()

        # init statics
        self._init_labels()
        self._load_fewshot_to_cls_weight()
        self.logger.info("Initialization done!")

    def _define_task(self, config):
        # specify task
        self.fewshot = config.data_params.fewshot
        self.clus = config.cluster is not None
        self.cls = self.semi_src = self.ssl = False
        self.is_pseudo_src = self.is_pseudo_tgt = False

        for ls in config.loss_params.loss:
            self.cls = self.cls | ls.startswith("cls")
            self.semi_src = self.semi_src | ls.startswith("semi")
            self.ssl = self.ssl | (ls.split("-")[0] not in ["cls", "semi", "tgt", "sim", "shp", "aux"])
            self.is_pseudo_src = self.is_pseudo_src | ls.startswith("semi-pseudo")
            self.is_pseudo_tgt = self.is_pseudo_tgt | ls.startswith("tgt-pseudo")

        self.is_pseudo_src = self.is_pseudo_src | (config.loss_params.pseudo and self.fewshot is not None)
        self.is_pseudo_tgt = self.is_pseudo_tgt | config.loss_params.pseudo
        self.semi_src = self.semi_src | self.is_pseudo_src

        if self.clus:
            self.is_pseudo_tgt = self.is_pseudo_tgt | (config.cluster.tgt_GC == "PGC" and "GC" in config.cluster.type)

    def _init_labels(self):
        train_len_src = self.get_attr("source", "train_len")
        train_len_tgt = self.get_attr("target", "train_len")

        # labels for pseudo
        if self.fewshot:
            self.predict_ordered_labels_pseudo_source = (torch.zeros(train_len_src, dtype=torch.long).detach().cuda() - 1)
            for ind, lbl in zip(self.fewshot_index_source, self.fewshot_label_source):
                self.predict_ordered_labels_pseudo_source[ind] = lbl

        self.predict_ordered_labels_pseudo_target = (torch.zeros(train_len_tgt, dtype=torch.long).detach().cuda() - 1)

    def _load_datasets(self):
        data_root = self.config.data_params.data_root
        num_workers = self.config.data_params.num_workers
        fewshot = self.config.data_params.fewshot
        data_aug = self.config.data_params.data_aug
        griding_num = self.griding_num
        num_lanes = self.num_lanes
        image_height = self.config.data_params.image_height_net
        image_width = self.config.data_params.image_width_net
        row_anchor = [x * image_height for x in self.h_samples]
        use_aux = self.use_aux

        list_path_train = {
            "source": self.config.data_params.source_train,
            "target": self.config.data_params.target_train
        }

        list_path_target_train_pseudo = self.config.data_params.target_train_pseudo

        list_path_val = {
            "source": self.config.data_params.source_val,
            "target": self.config.data_params.target_val
        }

        list_path_test = self.config.data_params.target_test

        batch_size_dict = {
            "val": self.config.optim_params.batch_size,
            "source": self.config.optim_params.batch_size_src,
            "target": self.config.optim_params.batch_size_tgt,
            "labeled": self.config.optim_params.batch_size_lbd,
        }

        self.batch_size_dict = batch_size_dict

        # Unlabeled Train Dataset for Self-supervised Learning
        for domain_name in ("source", "target"):
            train_dataset = datautils.create_dataset_unl(data_root,
                                                         list_path_train[domain_name],
                                                         image_height=image_height,
                                                         image_width=image_width,
                                                         data_aug=data_aug)

            train_loader = datautils.create_loader(train_dataset, batch_size_dict[domain_name], is_train=True, num_workers=num_workers)
            train_loader_init_mb = datautils.create_loader(train_dataset, batch_size_dict[domain_name], is_train=False, num_workers=num_workers)
            # train_labels = torch.from_numpy(train_dataset.labels).detach().cuda()

            self.set_attr(domain_name, "train_dataset", train_dataset)
            # self.set_attr(domain_name, "train_ordered_labels", train_labels)
            self.set_attr(domain_name, "train_loader", train_loader)
            self.set_attr(domain_name, "train_loader_init_mb", train_loader_init_mb)
            self.set_attr(domain_name, "train_len", len(train_dataset))

        # Few-shot Dataset
        if fewshot:
            # !!! fewshot learning is not supported !!!
            # get_fewshot_index does not work with lane detection dataset
            train_lbd_dataset_source = datautils.create_dataset_lbd(data_root,
                                                                    list_path_train["source"],
                                                                    image_height=image_height,
                                                                    image_width=image_width,
                                                                    griding_num=griding_num,
                                                                    row_anchor=row_anchor,
                                                                    num_lanes=num_lanes,
                                                                    use_aux=use_aux)
            src_dataset = self.get_attr("source", "train_dataset")
            (self.fewshot_index_source, self.fewshot_label_source) = datautils.get_fewshot_index(train_lbd_dataset_source, src_dataset)

            val_unl_dataset_source = datautils.create_dataset_unl(data_root,
                                                                  list_path_val["source"],
                                                                  image_height=image_height,
                                                                  image_width=image_width)

            self.val_unl_loader_source = datautils.create_loader(val_unl_dataset_source, batch_size_dict["val"], is_train=False, num_workers=num_workers)

            # labels for fewshot
            train_len = self.get_attr("source", "train_len")
            self.fewshot_labels = (torch.zeros(train_len, dtype=torch.long).detach().cuda() - 1)
            for ind, lbl in zip(self.fewshot_index_source, self.fewshot_label_source):
                self.fewshot_labels[ind] = lbl
        else:
            # Labeled Train Dataset Source for cls loss
            train_lbd_dataset_source = datautils.create_dataset_lbd(data_root,
                                                                    list_path_train["source"],
                                                                    image_height=image_height,
                                                                    image_width=image_width,
                                                                    griding_num=griding_num,
                                                                    row_anchor=row_anchor,
                                                                    num_lanes=num_lanes,
                                                                    use_aux=use_aux,
                                                                    data_aug=data_aug)

        self.train_lbd_loader_source = datautils.create_loader(train_lbd_dataset_source, batch_size_dict["labeled"], num_workers=num_workers)

        # Labeled Train Dataset Target for pseudo loss
        if self.is_pseudo_tgt:
            train_pseudo_dataset_target = datautils.create_dataset_pseudo(data_root,
                                                                          list_path_target_train_pseudo,
                                                                          image_height=image_height,
                                                                          image_width=image_width,
                                                                          griding_num=griding_num,
                                                                          row_anchor=row_anchor,
                                                                          num_lanes=num_lanes,
                                                                          use_aux=use_aux,
                                                                          data_aug=data_aug)

            self.train_pseudo_loader_target = datautils.create_loader(train_pseudo_dataset_target, batch_size_dict["labeled"], num_workers=num_workers)

        # Validation Dataset
        for domain_name in ("source", "target"):
            val_dataset = datautils.create_dataset_lbd(data_root,
                                                       list_path_val[domain_name],
                                                       image_height=image_height,
                                                       image_width=image_width,
                                                       griding_num=griding_num,
                                                       row_anchor=row_anchor,
                                                       num_lanes=num_lanes,
                                                       use_aux=use_aux)

            val_loader = datautils.create_loader(val_dataset, batch_size_dict["val"], is_train=False, num_workers=num_workers)
            self.set_attr(domain_name, "val_loader", val_loader)

        # Test Dataset
        test_dataset = datautils.create_dataset_lbd(data_root,
                                                    list_path_test,
                                                    image_height=image_height,
                                                    image_width=image_width,
                                                    griding_num=griding_num,
                                                    row_anchor=row_anchor,
                                                    num_lanes=num_lanes,
                                                    use_aux=use_aux)

        test_loader = datautils.create_loader(test_dataset, batch_size_dict["val"], is_train=False, num_workers=num_workers)
        self.set_attr("target", "test_loader", test_loader)

    def _create_model(self):
        backbone = self.config.model_params.backbone.split("-")[1]
        use_pretrained_cls = True if self.config.model_params.backbone.split("-")[-1] == "cls" else False
        cls_dim = self.cls_dim
        use_aux = self.use_aux
        pretrained = self.config.model_params.pretrained
        temp = self.config.model_params.cls_temp

        # ImageNet pretrained model
        if isinstance(pretrained, bool):
            model = Net(backbone=backbone, cls_dim=cls_dim, use_aux=use_aux, pretrained=pretrained)
            if pretrained:
                self.logger.info("ImageNet pretrained model used")
            else:
                self.logger.info("No pretrained ImageNet model used")

        # UFLD pretrained model
        elif isinstance(pretrained, str) and self.config.pretrained_exp_dir is None:
            model = Net(backbone=backbone, cls_dim=cls_dim, use_aux=use_aux, pretrained=False)
            pretrained_state_dict = torch.load(pretrained, map_location="cpu")["model"]
            pretrained_state_dict = {(key.replace("module.", "") if "module." in key else key): value for key, value in pretrained_state_dict.items()}

            # only use backbone parameters (feature encoder and auxiliary segmentation)
            new_state_dict = {key: value for key, value in pretrained_state_dict.items() if "model" in key or "aux" in key or "pool" in key}
            model.load_state_dict(new_state_dict, strict=False)
            self.logger.info("Pretrained encoder loaded")
            self.logger.info(f"{pretrained} loaded")

        self.model = nn.DataParallel(model, device_ids=self.gpu_devices).cuda()

        # Classification head
        if self.cls:
            if not use_pretrained_cls:
                cls_head = CosineClassifier(cls_dim=cls_dim, temp=temp)
                self.cls_head = nn.DataParallel(cls_head, device_ids=self.gpu_devices).cuda()
                self.logger.info("Using cosine classifier")

            elif use_pretrained_cls and isinstance(pretrained, str) and self.config.pretrained_exp_dir is None:
                cls_head = Classifier(cls_dim=cls_dim)
                cls_state_dict = {key: value for key, value in pretrained_state_dict.items() if "cls" in key}
                cls_head.load_state_dict(cls_state_dict, strict=False)
                self.cls_head = nn.DataParallel(cls_head, device_ids=self.gpu_devices).cuda()
                self.logger.info("Pretrained classifier loaded")
            else:
                raise NotImplementedError("Classifier could not be initialized")

        self.logger.info(f"Params of Model: {torchutils.model_params_num(self.model):,}")
        self.logger.info(f"Params of Classifier: {torchutils.model_params_num(self.cls_head):,}")

    def _create_optimizer(self):
        lr = self.config.optim_params.learning_rate
        momentum = self.config.optim_params.momentum
        nesterov = self.config.optim_params.nesterov
        weight_decay = self.config.optim_params.weight_decay
        conv_lr_ratio = self.config.optim_params.conv_lr_ratio
        optimizer = self.config.optim_params.optimizer
        scheduler = self.config.optim_params.scheduler
        cls_update = self.config.optim_params.cls_update
        milestones = self.config.optim_params.milestones
        gamma = self.config.optim_params.gamma
        num_epochs = self.config.num_epochs

        if self.config.steps_epoch is None:
            source_loader = self.get_attr("source", "train_loader")
            target_loader = self.get_attr("target", "train_loader")
            steps_epoch = max(len(source_loader), len(target_loader))
        else:
            steps_epoch = self.config.steps_epoch

        # Parameters
        parameters = torchutils.get_parameters(self.model, self.cls_head, self.cls, cls_update, conv_lr_ratio, lr)

        # Optimizer
        if optimizer == "Adam":
            self.optim = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer == "SGD":
            self.optim = torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        else:
            raise NotImplementedError

        # Scheduler
        if scheduler == "multi":
            multi = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=milestones, gamma=gamma)
            self.lr_scheduler_list.append(multi)
        elif scheduler == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, num_epochs * steps_epoch * 2, eta_min=0)
        elif scheduler == "lambda":
            self.lr_scheduler = torchutils.lambda_scheduler(self.optim)
        else:
            raise NotImplementedError

    def train_one_epoch(self):
        # train preparation
        self.model.train()
        if self.cls:
            self.cls_head.train()
        self.loss_fn.module.epoch = self.current_epoch

        loss_list = [loss for loss in self.config.loss_params.loss]
        loss_weight = [weight for weight in self.config.loss_params.loss.values()]
        loss_warmup = self.config.loss_params.start
        loss_giveup = self.config.loss_params.end

        num_loss = len(loss_list)

        source_loader = self.get_attr("source", "train_loader")
        target_loader = self.get_attr("target", "train_loader")

        if self.config.steps_epoch is None:
            num_batches = max(len(source_loader), len(target_loader))
            self.logger.info(f"source loader batches: {len(source_loader)}")
            self.logger.info(f"target loader batches: {len(target_loader)}")
        else:
            num_batches = self.config.steps_epoch

        epoch_loss = AverageMeter()
        epoch_loss_parts = [AverageMeter() for _ in range(num_loss)]

        # stop self-supervised learning after n epochs
        if self.ssl and self.current_epoch > self.config.loss_params.ssl_end_after:
            self.ssl = False
            self.logger.info(f"Turning off self-supervised losses")

        # cluster
        if self.clus:
            if self.config.cluster.kmeans_freq:
                kmeans_batches = num_batches // self.config.cluster.kmeans_freq
            else:
                kmeans_batches = 1
        else:
            kmeans_batches = None

        # load weight
        self._load_fewshot_to_cls_weight()

        if self.fewshot:
            fewshot_index = torch.tensor(self.fewshot_index_source).cuda()

        tqdm_batch = tqdm(total=num_batches, desc=f"[Epoch {self.current_epoch}]", leave=False)
        tqdm_post = {}

        for batch_i in range(num_batches):
            # Kmeans
            if is_div(kmeans_batches, batch_i) and self.ssl:
                self._update_cluster_labels()

            if not self.config.optim_params.cls_update:
                self._load_fewshot_to_cls_weight()

            # iteration over all source images
            if not batch_i % len(source_loader):
                source_iter = iter(source_loader)

                if "semi-condentmax" in loss_list:
                    momentum_prob_source = (self.momentum_softmax_source.softmax_vector.cuda())
                    self.momentum_softmax_source.reset()

            # iteration over all target images
            if not batch_i % len(target_loader):
                target_iter = iter(target_loader)

                if "tgt-condentmax" in loss_list:
                    momentum_prob_target = (self.momentum_softmax_target.softmax_vector.cuda())
                    self.momentum_softmax_target.reset()

            # iteration over all labeled source images
            if self.cls and not batch_i % len(self.train_lbd_loader_source):
                source_lbd_iter = iter(self.train_lbd_loader_source)

            if self.is_pseudo_tgt and not batch_i % len(self.train_pseudo_loader_target):
                target_pseudo_iter = iter(self.train_pseudo_loader_target)

            # calculate loss
            for domain_name in ("source", "target"):
                loss = torch.tensor(0).cuda()
                loss_d = 0
                loss_part_d = [0] * num_loss
                batch_size = self.batch_size_dict[domain_name]

                # inference for labeled source data
                if self.cls and domain_name == "source":
                    if self.use_aux:
                        indices_lbd, images_lbd, labels_lbd, seg_lbd = next(source_lbd_iter)
                    else:
                        indices_lbd, images_lbd, labels_lbd = next(source_lbd_iter)
                    indices_lbd = indices_lbd.cuda()
                    images_lbd = images_lbd.cuda()
                    labels_lbd = labels_lbd.long().cuda()
                    if self.use_aux:
                        seg_lbd = seg_lbd.long().cuda()
                        feat_lbd, seg_out_lbd = self.model(images_lbd)
                    else:
                        feat_lbd = self.model(images_lbd)
                    if self.use_l2_norm:
                        feat_lbd = F.normalize(feat_lbd, dim=1)
                    out_lbd = self.cls_head(feat_lbd)

                # inference for labeled target data
                if self.is_pseudo_tgt and domain_name == 'target':
                    if self.use_aux:
                        indices_lbd, images_lbd, labels_lbd, seg_lbd, confidence_cls = next(target_pseudo_iter)
                    else:
                        indices_lbd, images_lbd, labels_lbd, confidence_cls = next(target_pseudo_iter)
                    indices_lbd = indices_lbd.cuda()
                    images_lbd = images_lbd.cuda()
                    labels_lbd = labels_lbd.long().cuda()
                    confidence_cls = confidence_cls.cuda()
                    if self.use_aux:
                        seg_lbd = seg_lbd.long().cuda()
                        feat_lbd, seg_out_lbd = self.model(images_lbd)
                    else:
                        feat_lbd = self.model(images_lbd)
                    if self.use_l2_norm:
                        feat_lbd = F.normalize(feat_lbd, dim=1)
                    out_lbd = self.cls_head(feat_lbd)

                # Matching & ssl, inference for all source or target data
                if ("tgt-condentmax" in loss_list or "tgt-entmin" in loss_list) or self.ssl:
                    loader_iter = (source_iter if domain_name == "source" else target_iter)
                    indices_unl, images_unl = next(loader_iter)
                    indices_unl = indices_unl.cuda()
                    images_unl = images_unl.cuda()
                    if self.use_aux:
                        feat_unl, _ = self.model(images_unl)
                    else:
                        feat_unl = self.model(images_unl)
                    if self.use_l2_norm:
                        feat_unl = F.normalize(feat_unl, dim=1)
                    out_unl = self.cls_head(feat_unl)

                # Semi Supervised
                if self.semi_src and domain_name == "source":
                    semi_mask = ~torchutils.isin(indices_unl, fewshot_index)
                    indices_semi = indices_unl[semi_mask]
                    out_semi = out_unl[semi_mask]

                # Self-supervised Learning
                if self.ssl:
                    _, new_data_memory, loss_ssl, aux_list = self.loss_fn(indices_unl, feat_unl, domain_name, self.parallel_helper_idxs)
                    loss_ssl = [torch.mean(ls) for ls in loss_ssl]

                # fewshot memory bank
                if domain_name == "source" and self.fewshot:
                    mb = self.get_attr("source", "memory_bank_wrapper")
                    indices_lbd_tounl = fewshot_index[indices_lbd]
                    mb_feat_lbd = mb.at_idxs(indices_lbd_tounl)
                    fewshot_data_memory = update_data_memory(mb_feat_lbd, feat_lbd)

                # Pseudo-Labeling
                """
                thres_dict = {
                    "source": self.config.loss_params.thres_src,
                    "target": self.config.loss_params.thres_tgt,
                }

                # loss_pseudo = torch.tensor(0).cuda()
                is_pseudo = {
                    "source": self.is_pseudo_src,
                    "target": self.is_pseudo_tgt
                }

                if is_pseudo[domain_name]:
                    if domain_name == "source":
                        indices_pseudo = indices_semi
                        out_pseudo = out_semi
                        pseudo_domain = self.predict_ordered_labels_pseudo_source
                    else:
                        indices_pseudo = indices_unl
                        out_pseudo = out_unl  # [bs, class_num]
                        pseudo_domain = self.predict_ordered_labels_pseudo_target

                    thres = thres_dict[domain_name]

                    # calculate loss
                    loss_pseudo, aux = torchutils.pseudo_label_loss(out_pseudo, thres=thres, mask=None, num_class=self.num_class, aux=True)
                    mask_pseudo = aux["mask"]

                    # stat
                    pred_selected = out_pseudo.argmax(dim=1)[mask_pseudo]
                    indices_selected = indices_pseudo[mask_pseudo]
                    indices_unselected = indices_pseudo[~mask_pseudo]

                    pseudo_domain[indices_selected] = pred_selected
                    pseudo_domain[indices_unselected] = -1
                """

                # Compute Loss
                for ind, ls in enumerate(loss_list):
                    if self.current_epoch < loss_warmup[ind] or self.current_epoch >= loss_giveup[ind]:
                        continue

                    loss_part = torch.tensor(0).cuda()

                    # classification on labeled source domain
                    if ls == "cls-so" and domain_name == "source":
                        loss_part = self.loss_cls(out_lbd, labels_lbd)

                    # classification on few-shot
                    elif ls == "cls-info" and domain_name == "source" and self.fewshot:
                        loss_part = loss_info(feat_lbd, mb_feat_lbd, labels_lbd)

                    # similarity loss
                    elif ls == "sim-src" and domain_name == "source":
                        loss_part = self.loss_sim(out_lbd)
                    elif ls == "sim-tgt" and domain_name == "target" and self.is_pseudo_tgt:
                        valid_target = confidence_cls >= self.config.loss_params.thres_tgt
                        valid_mask_target = valid_target.nonzero(as_tuple=False)[:, 0]
                        loss_part = self.loss_sim(out_lbd[valid_mask_target])

                    # shape loss
                    elif ls == "shape-src" and domain_name == "source":
                        loss_part = self.loss_shape(out_lbd)
                    elif ls == "shape-tgt" and domain_name == "target" and self.is_pseudo_tgt:
                        valid_target = confidence_cls >= self.config.loss_params.thres_tgt
                        valid_mask_target = valid_target.nonzero(as_tuple=False)[:, 0]
                        loss_part = self.loss_shape(out_lbd[valid_mask_target])

                    # auxiliary loss
                    elif ls == "aux-src" and domain_name == "source" and self.use_aux:
                        loss_part = self.loss_aux(seg_out_lbd, seg_lbd)
                    elif ls == "aux-tgt" and domain_name == "target" and self.use_aux and self.is_pseudo_tgt:
                        valid_target = confidence_cls >= self.config.loss_params.thres_tgt
                        valid_mask_target = valid_target.nonzero(as_tuple=False)[:, 0]
                        loss_part = self.loss_aux(seg_out_lbd[valid_mask_target], seg_lbd[valid_mask_target])

                    # semi-supervision learning on unlabeled source
                    elif ls == "semi-entmin" and domain_name == "source":
                        loss_part = torchutils.entropy(out_semi, self.num_class)
                    elif ls == "semi-condentmax" and domain_name == "source":
                        bs = out_semi.size(0)
                        prob_semi = F.softmax(out_semi, dim=1)
                        prob_semi = prob_semi.view(-1, self.num_class)
                        prob_mean_semi = prob_semi.sum(dim=0) / bs

                        # update momentum
                        self.momentum_softmax_source.update(prob_mean_semi.cpu().detach(), bs)

                        # get momentum probability
                        momentum_prob_source = (self.momentum_softmax_source.softmax_vector.cuda())

                        # compute loss
                        entropy_cond = -torch.sum(prob_mean_semi * torch.log(momentum_prob_source + 1e-5))
                        loss_part = -entropy_cond

                    # learning on unlabeled target domain
                    elif ls == "tgt-entmin" and domain_name == "target":
                        loss_part = torchutils.entropy(out_unl, self.num_class)
                    elif ls == "tgt-condentmax" and domain_name == "target":
                        bs = out_unl.size(0)
                        prob_unl = F.softmax(out_unl, dim=1)
                        prob_unl = prob_unl.view(-1, self.num_class)
                        prob_mean_unl = prob_unl.sum(dim=0) / bs

                        # update momentum
                        self.momentum_softmax_target.update(prob_mean_unl.cpu().detach(), bs)

                        # get momentum probability
                        momentum_prob_target = (self.momentum_softmax_target.softmax_vector.cuda())

                        # compute loss
                        entropy_cond = -torch.sum(prob_mean_unl * torch.log(momentum_prob_target + 1e-5))
                        loss_part = -entropy_cond

                    # pseudo labeling loss
                    elif ls == "semi-pseudo" and domain_name == "source" and self.is_pseudo_src:
                        valid_source = confidence_cls >= self.config.loss_params.thres_src
                        valid_mask_source = valid_source.nonzero(as_tuple=False)[:, 0]
                        loss_part = self.loss_pseudo(out_lbd[valid_mask_source], labels_lbd[valid_mask_source])
                    elif ls == "tgt-pseudo" and domain_name == "target" and self.is_pseudo_tgt:
                        valid_target = confidence_cls >= self.config.loss_params.thres_tgt
                        valid_mask_target = valid_target.nonzero(as_tuple=False)[:, 0]
                        loss_part = self.loss_pseudo(out_lbd[valid_mask_target], labels_lbd[valid_mask_target])

                    # self-supervised learning
                    elif ls.split("-")[0] in ["ID", "CD", "proto", "I2C", "C2C"] and self.ssl:
                        loss_part = loss_ssl[ind]

                    loss_part = loss_weight[ind] * loss_part
                    loss = loss + loss_part
                    loss_d = loss_d + loss_part.item()
                    loss_part_d[ind] = loss_part.item()

                # Backpropagation
                self.optim.zero_grad()
                if len(loss_list) and loss != 0:
                    loss.backward()
                self.optim.step()

                # update memory_bank after backpropagation
                if self.ssl:
                    self._update_memory_bank(domain_name, indices_unl, new_data_memory)
                    if domain_name == "source" and self.fewshot:
                        self._update_memory_bank(domain_name, indices_lbd_tounl, fewshot_data_memory)

                # update lr info
                tqdm_post["lr"] = torchutils.get_lr(self.optim, g_id=-1)

                # update loss info
                epoch_loss.update(loss_d, batch_size)
                tqdm_post["loss"] = epoch_loss.avg
                self.summary_writer.add_scalars("train/loss", {"total": epoch_loss.val}, self.current_iteration)
                self.summary_writer.add_scalars("train/lr", {"total": torchutils.get_lr(self.optim, g_id=-1)}, self.current_iteration)
                wandb.log({"train/loss": epoch_loss.val})
                wandb.log({"train/lr": torchutils.get_lr(self.optim, g_id=-1)})
                self.train_loss.append(epoch_loss.val)

                # update loss part info
                domain_iteration = self.get_attr(domain_name, "current_iteration")
                self.summary_writer.add_scalars(f"train/{self.domain_map[domain_name]}_loss", {"total": epoch_loss.val}, domain_iteration)
                wandb.log({f"train/{self.domain_map[domain_name]}_loss": epoch_loss.val})

                loss_dict = {}
                for i, ls in enumerate(loss_part_d):
                    ls_name = loss_list[i]
                    epoch_loss_parts[i].update(ls, batch_size)
                    tqdm_post[ls_abbr[ls_name]] = epoch_loss_parts[i].avg
                    self.summary_writer.add_scalars(f"train/{self.domain_map[domain_name]}_loss", {ls_name: epoch_loss_parts[i].val}, domain_iteration)
                    loss_dict[ls_name] = epoch_loss_parts[i].val

                wandb.log({f"train/{self.domain_map[domain_name]}_loss": loss_dict})

                # Logging images with weights and biases
                if not batch_i % 5000 and self.ssl:
                    if domain_name == "source":
                        wandb.log({f"{domain_name} images labeled": [wandb.Image(image.permute(1, 2, 0).detach().cpu().numpy()) for image in images_lbd]})
                        wandb.log({f"{domain_name} images unlabeled": [wandb.Image(image.permute(1, 2, 0).detach().cpu().numpy()) for image in images_unl]})
                    if domain_name == "target":
                        wandb.log({f"{domain_name} images unlabeled": [wandb.Image(image.permute(1, 2, 0).detach().cpu().numpy()) for image in images_unl]})

                # adjust lr
                if self.config.optim_params.scheduler == "cosine" or self.config.optim_params.scheduler == "lambda":
                    self.lr_scheduler.step()

                self.current_iteration += 1

            tqdm_batch.set_postfix(tqdm_post)
            tqdm_batch.update()
            self.current_iteration_source += 1
            self.current_iteration_target += 1
        tqdm_batch.close()

        self.current_loss = epoch_loss.avg

    @torch.no_grad()
    def _load_fewshot_to_cls_weight(self):
        """
        Load centroids to cosine classifier (Adaptive Prototype-Classifier Update)

        This part is not possible with multi-class classification

        Args:
            method (str, optional): None, "fewshot", "src", "tgt". Defaults to None.
        """

        method = self.config.model_params.load_weight

        if method is None:
            return
        assert method in ["fewshot", "src", "tgt", "src-tgt", "fewshot-tgt"]

        thres = {
            "src": 1,
            "tgt": self.config.model_params.load_weight_thres
        }
        bank = {
            "src": self.get_attr("source", "memory_bank_wrapper").as_tensor(),
            "tgt": self.get_attr("target", "memory_bank_wrapper").as_tensor(),
        }

        fewshot_label = {}
        fewshot_index = {}

        is_tgt = method in ["tgt", "fewshot-tgt", "src-tgt"] and self.current_epoch >= self.config.model_params.load_weight_epoch

        if method in ["fewshot", "fewshot-tgt"]:
            if self.fewshot:
                fewshot_label["src"] = torch.tensor(self.fewshot_label_source)
                fewshot_index["src"] = torch.tensor(self.fewshot_index_source)
            else:
                fewshot_label["src"] = self.get_attr("source", "train_ordered_labels")
                fewshot_index["src"] = torch.arange(self.get_attr("source", "train_len"))
        else:
            mask = self.predict_ordered_labels_pseudo_source != -1
            fewshot_label["src"] = self.predict_ordered_labels_pseudo_source[mask]
            fewshot_index["src"] = mask.nonzero().squeeze(1)

        if is_tgt:
            # pseudo labels target
            mask = self.predict_ordered_labels_pseudo_target != -1
            fewshot_label["tgt"] = self.predict_ordered_labels_pseudo_target[mask]
            fewshot_index["tgt"] = mask.nonzero().squeeze(1)

        for domain in ("src", "tgt"):
            if domain == "tgt" and not is_tgt:
                break
            if domain == "src" and method == "tgt":
                break
            weight = self.cls_head[2].weight.data

            for label in range(self.num_class):
                fewshot_mask = fewshot_label[domain] == label
                if fewshot_mask.sum() < thres[domain]:
                    continue
                fewshot_ind = fewshot_index[domain][fewshot_mask]
                bank_vec = bank[domain][fewshot_ind]
                weight[label] = F.normalize(torch.mean(bank_vec, dim=0), dim=0)

    # Validate

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        # Domain Adaptation
        if self.cls:
            # self._load_fewshot_to_cls_weight()
            self.cls_head.eval()
            if self.config.data_params.fewshot:
                self.score(self.val_unl_loader_source, "source", name=f"{self.domain_map['source']}_val")

            val_loader_source = self.get_attr("source", "val_loader")
            val_loader_target = self.get_attr("target", "val_loader")
            test_loader_target = self.get_attr("target", "test_loader")

            self.score(val_loader_source, "source", name=f"{self.domain_map['source']}_val")
            self.score(val_loader_target, "target", name=f"{self.domain_map['target']}_val")
            self.current_val_metric = self.score(test_loader_target, "target", name=f"{self.domain_map['target']}_test")

        # update information
        self.current_val_iteration += 1
        if self.current_val_metric >= self.best_val_metric:
            self.best_val_metric = self.current_val_metric
            self.best_val_epoch = self.current_epoch
            self.iter_with_no_improv = 0
        else:
            self.iter_with_no_improv += 1
        self.val_acc.append(self.current_val_metric)

        self.clear_train_features()

    @torch.no_grad()
    def score(self, loader, domain, name="val"):
        results = {}
        epoch_loss = AverageMeter()

        reset_metrics(self.metric_dict)

        if self.use_aux:
            tqdm_batch = tqdm(total=len(loader), desc=f"[Validate model on {domain}]")
            for batch_i, (indices, images, labels, seg_labels) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()
                seg_labels = seg_labels.cuda()

                feat, seg_out = self.model(images)
                if self.use_l2_norm:
                    feat = F.normalize(feat, dim=1)
                output = self.cls_head(feat)

                loss = self.loss_cls(output, labels)

                results["cls_out"] = torch.argmax(output, dim=1)
                results["cls_label"] = labels
                results["seg_out"] = torch.argmax(seg_out, dim=1)
                results["seg_label"] = seg_labels

                update_metrics(self.metric_dict, results, use_aux=True)
                epoch_loss.update(loss, results["cls_out"].size(0))

                for me_name, me_op in zip(self.metric_dict["name"], self.metric_dict["op"]):
                    self.summary_writer.add_scalars("val/acc_" + me_name, {f"{name}": me_op.get()}, self.current_epoch)
                    wandb.log({f"val/acc_{me_name}_{name}": me_op.get()})

                tqdm_batch.update()
            tqdm_batch.close()
        else:
            tqdm_batch = tqdm(total=len(loader), desc=f"[Validate model on {domain}]")
            for batch_i, (indices, images, labels) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()

                feat = self.model(images)
                if self.use_l2_norm:
                    feat = F.normalize(feat, dim=1)
                output = self.cls_head(feat)

                loss = self.loss_cls(output, labels)

                results["cls_out"] = torch.argmax(output, dim=1)
                results["cls_label"] = labels

                update_metrics(self.metric_dict, results, use_aux=False)
                epoch_loss.update(loss, results["cls_out"].size(0))

                for me_name, me_op in zip(self.metric_dict["name"][:3], self.metric_dict["op"][:3]):
                    self.summary_writer.add_scalars("val/acc_" + me_name, {f"{name}": me_op.get()}, self.current_epoch)
                    wandb.log({f"val/acc_{me_name}_{name}": me_op.get()})

                tqdm_batch.update()
            tqdm_batch.close()

        acc = self.metric_dict["op"][0].get()
        if self.use_aux:
            metrics = [f"{me_name}={100. * me_op.get():.3f}%" for me_name, me_op in zip(self.metric_dict["name"], self.metric_dict["op"])]
        else:
            metrics = [f"{me_name}={100. * me_op.get():.3f}%" for me_name, me_op in zip(self.metric_dict["name"][:3], self.metric_dict["op"][:3])]

        self.summary_writer.add_scalars("val/loss", {f"{name}": epoch_loss.avg}, self.current_epoch)
        wandb.log({f"val/loss_{name}": epoch_loss.avg})
        self.logger.info(f"[Epoch {self.current_epoch} {name}] loss={epoch_loss.avg:.5f}, acc={metrics}")

        return acc

    # Load & Save checkpoint

    def load_checkpoint(self, filename, checkpoint_dir=None, load_memory_bank=False, load_model=True, load_optim=False, load_epoch=False, load_cls=True):
        checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        filename = os.path.join(checkpoint_dir, filename)
        try:
            self.logger.info(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename, map_location="cpu")

            if load_epoch:
                self.current_epoch = checkpoint["epoch"]
                for domain_name in ("source", "target"):
                    self.set_attr(domain_name, "current_iteration", checkpoint[f"iteration_{domain_name}"])
                self.current_iteration = checkpoint["iteration"]
                self.current_val_iteration = checkpoint["val_iteration"]

            if load_model:
                model_state_dict = checkpoint["model_state_dict"]
                self.model.load_state_dict(model_state_dict)

            if load_cls and self.cls and "cls_state_dict" in checkpoint:
                cls_state_dict = checkpoint["cls_state_dict"]
                self.cls_head.load_state_dict(cls_state_dict)

            if load_optim:
                optim_state_dict = checkpoint["optim_state_dict"]
                self.optim.load_state_dict(optim_state_dict)

                lr_pretrained = self.optim.param_groups[0]["lr"]
                lr_config = self.config.optim_params.learning_rate

                # Change learning rate
                if not lr_pretrained == lr_config:
                    for param_group in self.optim.param_groups:
                        param_group["lr"] = self.config.optim_params.learning_rate

            if self.ssl:
                self._init_memory_bank()
            if load_memory_bank or self.config.model_params.init_memory_bank is False:
                self._load_memory_bank({
                    "source": checkpoint["memory_bank_source"],
                    "target": checkpoint["memory_bank_target"],
                })

            self.logger.info(
                f"Checkpoint loaded successfully from '{filename}' at (epoch {checkpoint['epoch']}) at (iteration s:{checkpoint['iteration_source']} t:{checkpoint['iteration_target']}) with loss = {checkpoint['loss']}\nval acc = {checkpoint['val_acc']}\n")

        except OSError as e:
            self.logger.info(f"Checkpoint doesnt exists: [{filename}]")
            raise e

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        out_dict = {
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optim.state_dict(),
            "epoch": self.current_epoch,
            "iteration": self.current_iteration,
            "iteration_source": self.get_attr("source", "current_iteration"),
            "iteration_target": self.get_attr("target", "current_iteration"),
            "val_iteration": self.current_val_iteration,
            "val_acc": np.array(self.val_acc),
            "val_metric": self.current_val_metric,
            "loss": self.current_loss,
            "train_loss": np.array(self.train_loss),
        }
        if self.config.model_params.save_mem_bank:
            out_dict["memory_bank_source"] = self.get_attr("source", "memory_bank_wrapper")
            out_dict["memory_bank_target"] = self.get_attr("target", "memory_bank_wrapper")
        if self.cls:
            out_dict["cls_state_dict"] = self.cls_head.state_dict()
        # best according to source-to-target
        is_best = (self.current_val_metric == self.best_val_metric) or not self.config.validate_freq
        torchutils.save_checkpoint(out_dict, is_best, filename=filename, folder=self.config.checkpoint_dir)
        self.copy_checkpoint()

    # compute train features

    @torch.no_grad()
    def compute_train_features(self):
        if self.is_features_computed:
            return
        else:
            self.is_features_computed = True
        self.model.eval()

        for domain in ("source", "target"):
            train_loader_init_mb = self.get_attr(domain, "train_loader_init_mb")
            imgs, features, idx = [], [], []
            tqdm_batch = tqdm(total=len(train_loader_init_mb), desc=f"[Compute train features of {domain}]")

            for batch_i, (indices, images) in enumerate(train_loader_init_mb):
                images = images.to(self.device)
                if self.use_aux:
                    feat, _ = self.model(images)
                else:
                    feat = self.model(images)
                if self.use_l2_norm:
                    feat = F.normalize(feat, dim=1)

                features.append(feat)
                idx.append(indices)

                tqdm_batch.update()
            tqdm_batch.close()

            features = torch.cat(features)
            idx = torch.cat(idx).to(self.device)

            self.set_attr(domain, "train_features", features)
            self.set_attr(domain, "train_indices", idx)

    def clear_train_features(self):
        self.is_features_computed = False

    # Memory bank

    @torch.no_grad()
    def _init_memory_bank(self):
        for domain_name in ("source", "target"):
            data_len = self.get_attr(domain_name, "train_len")
            memory_bank = MemoryBank(data_len, 1800)
            if self.config.model_params.init_memory_bank:
                self.compute_train_features()
                idx = self.get_attr(domain_name, "train_indices")
                feat = self.get_attr(domain_name, "train_features")
                memory_bank.update(idx, feat)
                # self.logger.info(f"Initialize memorybank-{domain_name} with pretrained output features")

                # save space
                if self.config.data_params.name in ["carla"]:
                    delattr(self, f"train_indices_{domain_name}")
                    delattr(self, f"train_features_{domain_name}")

            memory_bank_space = size_of_tensor(memory_bank.as_tensor())
            self.logger.info(f"Memory bank size {domain_name}: {memory_bank_space}")

            self.set_attr(domain_name, "memory_bank_wrapper", memory_bank)

            self.loss_fn.module.set_attr(domain_name, "data_len", data_len)
            self.loss_fn.module.set_broadcast(domain_name, "memory_bank", memory_bank.as_tensor())

    @torch.no_grad()
    def _update_memory_bank(self, domain_name, indices, new_data_memory):
        memory_bank_wrapper = self.get_attr(domain_name, "memory_bank_wrapper")
        memory_bank_wrapper.update(indices, new_data_memory)
        updated_bank = memory_bank_wrapper.as_tensor()
        self.loss_fn.module.set_broadcast(domain_name, "memory_bank", updated_bank)

    def _load_memory_bank(self, memory_bank_dict):
        """
        Load memory bank from checkpoint

        Args:
            memory_bank_dict (dict): memory_bank dict of source and target domain
        """
        for domain_name in ("source", "target"):
            memory_bank = memory_bank_dict[domain_name]._bank.cuda()
            self.get_attr(domain_name, "memory_bank_wrapper")._bank = memory_bank
            self.loss_fn.module.set_broadcast(domain_name, "memory_bank", memory_bank)

    # Cluster

    @torch.no_grad()
    def _update_cluster_labels(self):
        k_list = self.config.k_list
        for clus_type in self.config.cluster.type:
            cluster_labels_domain = {}
            cluster_centroids_domain = {}
            cluster_phi_domain = {}

            # clustering for each domain
            if clus_type == "each":
                for domain_name in ("source", "target"):
                    memory_bank_tensor = self.get_attr(domain_name, "memory_bank_wrapper").as_tensor()

                    # clustering
                    cluster_labels, cluster_centroids, cluster_phi = torch_kmeans(k_list, memory_bank_tensor, seed=self.current_epoch + self.current_iteration)

                    cluster_labels_domain[domain_name] = cluster_labels
                    cluster_centroids_domain[domain_name] = cluster_centroids
                    cluster_phi_domain[domain_name] = cluster_phi

                self.cluster_each_labels_domain = cluster_labels_domain
                self.cluster_each_centroids_domain = cluster_centroids_domain
                self.cluster_each_phi_domain = cluster_phi_domain
            else:
                print(clus_type)
                raise NotImplementedError

            # update cluster to loss_fn
            for domain_name in ("source", "target"):
                self.loss_fn.module.set_broadcast(domain_name, f"cluster_labels_{clus_type}", cluster_labels_domain[domain_name])
                self.loss_fn.module.set_broadcast(domain_name, f"cluster_centroids_{clus_type}", cluster_centroids_domain[domain_name])

                if cluster_phi_domain:
                    self.loss_fn.module.set_broadcast(domain_name, f"cluster_phi_{clus_type}", cluster_phi_domain[domain_name])
