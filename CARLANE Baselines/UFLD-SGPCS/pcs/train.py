#!/usr/bin/env python3

import argparse
from pcs.agents.TrainAgent import TrainAgent
from pcs.utils.setup import check_pretrain_dir, process_config
from pcs.utils.utils import load_json, set_default


def adjust_config(config):
    set_default(config, "validate_freq", value=1)
    set_default(config, "copy_checkpoint_freq", value=50)
    set_default(config, "debug", value=False)
    set_default(config, "cuda", value=True)
    set_default(config, "gpu_device", value=None)
    set_default(config, "pretrained_exp_dir", value=None)
    set_default(config, "pretrained_pcs_ufld", value=None)
    set_default(config, "steps_epoch", value=None)

    # data_params
    set_default(config.data_params, "num_workers", value=4)
    set_default(config.data_params, "fewshot", value=None)
    set_default(config.data_params, "image_height", value=720)
    set_default(config.data_params, "image_width", value=1280)
    set_default(config.data_params, "image_height_net", value=288)
    set_default(config.data_params, "image_width_net", value=800)

    # model_params
    set_default(config.model_params, "cls_temp", value=0.05)
    set_default(config.model_params, "use_l2_norm", value=True)
    set_default(config.model_params, "load_weight", value=None)
    set_default(config.model_params, "load_weight_thres", value=5)
    set_default(config.model_params, "load_weight_epoch", value=5)
    set_default(config.model_params, "init_memory_bank", value=True)
    set_default(config.model_params, "save_mem_bank", value=True)
    set_default(config.model_params, "row_anchor_start", value=160)

    # loss_params
    num_loss = len(config.loss_params.loss)
    set_default(config.loss_params, "start", value=[0] * num_loss)
    set_default(config.loss_params, "end", value=[1000] * num_loss)
    if not isinstance(config.loss_params.temp, list):
        config.loss_params.temp = [config.loss_params.temp] * num_loss
    set_default(config.loss_params, "m", value=0.5)
    set_default(config.loss_params, "pseudo", value=False)
    set_default(config.loss_params, "thres_src", value=0.7)
    set_default(config.loss_params, "thres_tgt", value=0.7)

    # optim_params
    set_default(config.optim_params, "batch_size_src", callback="batch_size")
    set_default(config.optim_params, "batch_size_tgt", callback="batch_size")
    set_default(config.optim_params, "batch_size_lbd", callback="batch_size")
    set_default(config.optim_params, "patience", value=10)
    set_default(config.optim_params, "momentum", value=0.9)
    set_default(config.optim_params, "nesterov", value=True)
    set_default(config.optim_params, "gamma", value=0.1)
    set_default(config.optim_params, "milestones", value=[5, 8])
    set_default(config.optim_params, "cls_update", value=True)

    # clustering
    if config.cluster is not None:
        if config.cluster.type is None:
            config.cluster = None
        else:
            if not isinstance(config.cluster.type, list):
                config.cluster.type = [config.cluster.type]
            k = config.cluster.k
            n_k = config.cluster.n_k
            config.k_list = k * n_k
            config.cluster.n_kmeans = len(config.k_list)

    return config


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/molane.json", help="the path to the config")
    parser.add_argument("--exp_id", type=str, default=None)

    # Dataset
    parser.add_argument("--dataset", type=str, default=None, choices=["molane"], help="name of dataset")  # TODO
    parser.add_argument("--source", type=str, default=None, help="source domain")
    parser.add_argument("--target", type=str, default=None, help="target domain")
    parser.add_argument("--fewshot", type=str, default=None, help="number of labeled samples")

    # Model
    parser.add_argument("--net", type=str, default=None, help="which network to use")

    # Optim
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (default: 0.001)")

    # Hyper-parameter
    parser.add_argument("--seed", type=int, default=None, metavar="S", help="random seed (default: 1)")

    return parser


def update_config(config_json, args):
    if args.dataset:
        config_json["data_params"]["name"] = args.dataset
    if args.source:
        config_json["data_params"]["source"] = args.source
    if args.target:
        config_json["data_params"]["target"] = args.target
    if args.fewshot:
        config_json["data_params"]["fewshot"] = args.fewshot
    if args.exp_id:
        config_json["exp_id"] = args.exp_id
    elif args.source:
        config_json["exp_id"] = f"{args.source}->{args.target}:{args.fewshot}"
    if args.seed:
        config_json["seed"] = args.seed
    if args.lr:
        config_json["optim_params"]["learning_rate"] = args.lr


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    # load config
    config_json = load_json(args.config)
    update_config(config_json, args)

    # check pretrain directory
    pre_checkpoint_dir = check_pretrain_dir(config_json)

    # json to DotMap
    config = process_config(config_json)
    config = adjust_config(config)

    # create agent
    agent = TrainAgent(config)

    if pre_checkpoint_dir is not None:
        agent.load_checkpoint("model_best.pth.tar", pre_checkpoint_dir, load_model=True, load_optim=True, load_epoch=True, load_cls=True)
    elif config.pretrained_pcs_ufld is not None:
        agent.load_checkpoint("ep000.pth.tar", config.pretrained_pcs_ufld)

    try:
        agent.run()
        agent.finalise()
    except KeyboardInterrupt:
        pass
