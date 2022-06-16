import os
import datetime
import wandb
import torch
from model.model import CNN, Discriminator
from data.dataloader import get_loader_labeled, get_loader_pseudo_train
from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, update_metrics, reset_metrics
from utils.common import merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger


def inference_val(net, data_label, use_aux, domain):
    if domain == 'source':
        if use_aux:
            image, cls_label, seg_label = data_label
            image, cls_label, seg_label = image.cuda(), cls_label.long().cuda(), seg_label.long().cuda()
            feature, seg_out = net.encoder(image)
        else:
            image, cls_label = data_label
            image, cls_label = image.cuda(), cls_label.long().cuda()
            feature = net.encoder(image)
    else:
        image, cls_label = data_label
        image, cls_label = image.cuda(), cls_label.long().cuda()
        feature = net.encoder(image)

    cls_out = net.classifier(feature)

    if use_aux:
        results = {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out': seg_out, 'seg_label': seg_label}
    else:
        results = {'cls_out': cls_out, 'cls_label': cls_label}

    return results


def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


def calc_loss(l_dict, results, global_step):
    loss = 0

    for i in range(len(l_dict['name'])):
        data_src = l_dict['data_src'][i]
        datas = [results[src] for src in data_src]
        loss_cur = l_dict['op'][i](*datas)

        if global_step % 20 == 0:
            wandb.log({'loss/' + l_dict['name'][i]: loss_cur})
        loss += loss_cur * l_dict['weight'][i]
    return loss


def train(epoch, use_aux):
    source_cnn.eval()
    target_cnn.encoder.train()
    discriminator.train()

    num_batches = max(len(train_loader_source), len(train_loader_target))
    progress_bar = dist_tqdm(range(num_batches), desc=f"[Epoch {epoch}]")

    for batch_i in progress_bar:
        # iteration over all source images
        if not batch_i % len(train_loader_source):
            source_iter = iter(train_loader_source)

        # iteration over all target images
        if not batch_i % len(train_loader_target):
            target_iter = iter(train_loader_target)

        data_source = next(source_iter)
        data_target = next(target_iter)

        global_step = epoch * num_batches + batch_i

        if use_aux:
            source_image, _, _ = data_source
            source_image = source_image.cuda()

            target_image, target_label, target_seg_label, cls_conf, pseudo_domain_label, disc_conf = data_target
            target_image, target_label, target_seg_label = target_image.cuda(), target_label.long().cuda(), target_seg_label.long().cuda()
            cls_conf, pseudo_domain_label, disc_conf = cls_conf.cuda(), pseudo_domain_label.cuda(), disc_conf.cuda()

            source_feat, _ = source_cnn.encoder(source_image)
            target_feat, _ = target_cnn.encoder(target_image)
        else:
            source_image, _ = data_source
            source_image = source_image.cuda()

            target_image, target_label, cls_conf, pseudo_domain_label, disc_conf = data_target
            target_image, target_label = target_image.cuda(), target_label.long().cuda()
            cls_conf = cls_conf.cuda()
            pseudo_domain_label = pseudo_domain_label.cuda()
            disc_conf = disc_conf.cuda()

            source_feat = source_cnn.encoder(source_image)
            target_feat = target_cnn.encoder(target_image)

        # train discriminator
        adversary_pred_source = discriminator(source_feat)
        adversary_pred_target = discriminator(target_feat)
        adversary_pred = torch.cat((adversary_pred_source, adversary_pred_target), 0)

        # Prepare domain labels
        adversary_labels_source = torch.zeros(source_image.shape[0]).type(torch.LongTensor).cuda()
        adversary_labels_target = torch.ones(target_image.shape[0]).type(torch.LongTensor).cuda()
        adversary_label = torch.cat((adversary_labels_source, adversary_labels_target), 0)

        # Train discriminator
        optimizer_discriminator.zero_grad()
        adversary_loss = criterion(adversary_pred, adversary_label)
        adversary_loss.backward()
        optimizer_discriminator.step()
        # scheduler_discriminator.step(global_step)

        # Train target encoder
        target_feat = target_cnn.encoder(target_image)
        adversary_pred_target = discriminator(target_feat)
        pred_target = target_cnn.classifier(target_feat)
        adversary_labels_source = torch.zeros(target_feat.shape[0]).type(torch.LongTensor).cuda()
        mapping_loss = criterion(adversary_pred_target, adversary_labels_source)

        valid_source = (pseudo_domain_label == 0) & (cls_conf >= cfg.threshold_cls)
        valid_mask_source = valid_source.nonzero(as_tuple=False)[:, 0]
        valid_target = (pseudo_domain_label == 1) & (disc_conf <= cfg.threshold_disc) & (cls_conf >= cfg.threshold_cls)
        valid_mask_target = valid_target.nonzero(as_tuple=False)[:, 0]
        valid_indexes = torch.cat((valid_mask_source, valid_mask_target), 0)
        pseudo_loss = criterion(pred_target[valid_indexes], target_label[valid_indexes])
        loss = mapping_loss + cfg.pseudo_loss_w * pseudo_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(global_step)

        adversary_pred = torch.argmax(adversary_pred, dim=1)
        acc = (adversary_pred == adversary_label).float().mean()

        if global_step % 20 == 0:
            wandb.log({'loss/pseudo_adv_mapping': loss})
            wandb.log({'loss/adv_disc': adversary_loss})
            wandb.log({'lr/disc': optimizer_discriminator.param_groups[0]['lr']})
            wandb.log({'lr/target_encoder': optimizer.param_groups[0]['lr']})
            wandb.log({'train/disc': acc})

        if hasattr(progress_bar, 'set_postfix'):
            progress_bar.set_postfix(loss_m='%.3f' % float(mapping_loss), loss_adv='%.3f' % float(adversary_loss), acc='%.3f' % acc)


@torch.no_grad()
def validate(net, val_loader, use_aux, domain=""):
    net.eval()

    val_iter = iter(val_loader)
    domain_name = 'source' if 'sim' in domain else 'target'

    progress_bar = dist_tqdm(val_loader, desc=f"[Validate on {domain_name}]")
    for _ in progress_bar:
        reset_metrics(metric_dict)
        data_label = next(val_iter)
        results = inference_val(net, data_label, use_aux, domain_name)
        results = resolve_val_data(results, use_aux)
        update_metrics(metric_dict, results)

        for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
            wandb.log({'val/' + me_name + '_' + domain: me_op.get()})

        if hasattr(progress_bar, 'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(**kwargs)


@torch.no_grad()
def validate_discriminator(source_loader, target_loader, use_aux):
    source_cnn.eval()
    target_cnn.eval()
    discriminator.eval()

    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    num_batches = min(len(source_loader), len(target_loader))
    domain_correct = 0

    progress_bar = dist_tqdm(range(num_batches), desc="[Validate Disc]")
    for _ in progress_bar:
        if use_aux:
            source_image, _, _ = next(iter_source)
            source_image = source_image.cuda()

            target_image, _, _ = next(iter_target)
            target_image = target_image.cuda()
        else:
            source_image, _ = next(iter_source)
            source_image = source_image.cuda()

            target_image, _ = next(iter_target)
            target_image = target_image.cuda()

        source_feat = source_cnn.encoder(source_image)
        target_feat = target_cnn.encoder(target_image)
        adversary_feat = torch.cat((source_feat, target_feat), 0)

        # Prepare disc labels
        source_adversary_labels = torch.zeros(source_image.shape[0]).type(torch.LongTensor)
        target_adversary_labels = torch.ones(target_image.shape[0]).type(torch.LongTensor)
        adversary_label = torch.cat((source_adversary_labels, target_adversary_labels), 0).cuda()

        adversary_pred = discriminator(adversary_feat)
        adversary_pred = torch.argmax(adversary_pred, dim=1)

        domain_correct += adversary_pred.eq(adversary_label).sum().cpu()

        total = 2 * min(len(source_loader.dataset), len(target_loader.dataset))
        acc = domain_correct / total

        wandb.log({'val/discriminator': acc})

        if hasattr(progress_bar, 'set_postfix'):
            progress_bar.set_postfix({'acc': '%.3f' % acc, 'correct': domain_correct.item(), 'total': total})


@torch.no_grad()
def test(test_loader, use_aux):
    target_cnn.eval()
    discriminator.eval()

    test_iter = iter(test_loader)

    progress_bar = dist_tqdm(test_loader, desc="[Test]")
    for _ in progress_bar:
        reset_metrics(metric_dict)
        data_label = next(test_iter)
        results = inference_val(target_cnn, data_label, use_aux, 'target')
        results = resolve_val_data(results, use_aux)

        update_metrics(metric_dict, results)

        for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
            wandb.log({'test/' + me_name + '_real_test': me_op.get()})

        if hasattr(progress_bar, 'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(**kwargs)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()
    work_dir = get_work_dir(cfg)
    dist_print(work_dir)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)

    wandb.init(project="UFLD-SGADA", name="UFLD-SGADA", config=cfg, dir=cfg.log_path)

    # Load data
    train_loader_source = get_loader_labeled(cfg.batch_size, cfg.data_root, cfg.source_train, cfg.griding_num, cfg.use_aux, distributed, cfg.num_lanes, cfg.num_workers)
    train_loader_target = get_loader_pseudo_train(cfg.batch_size, cfg.data_root, cfg.target_train_pseudo, cfg.griding_num, cfg.use_aux, distributed, cfg.num_lanes, cfg.num_workers)
    val_loader_source = get_loader_labeled(cfg.batch_size, cfg.data_root, cfg.source_val, cfg.griding_num, cfg.use_aux, distributed, cfg.num_lanes, cfg.num_workers)
    val_loader_target = get_loader_labeled(cfg.batch_size, cfg.data_root, cfg.target_val, cfg.griding_num, cfg.use_aux, distributed, cfg.num_lanes, cfg.num_workers)
    test_loader_target = get_loader_labeled(cfg.batch_size, cfg.data_root, cfg.target_test, cfg.griding_num, cfg.use_aux, distributed, cfg.num_lanes, cfg.num_workers)

    # Prepare model
    cls_dim = (cfg.griding_num + 1, cfg.cls_num_per_lane, cfg.num_lanes)
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    source_cnn = CNN(backbone=cfg.backbone, cls_dim=cls_dim, use_aux=cfg.use_aux).cuda()
    target_cnn = CNN(backbone=cfg.backbone, cls_dim=cls_dim, use_aux=cfg.use_aux, target=True, src_train=False).cuda()
    discriminator = Discriminator(slope=cfg.slope).cuda()

    if distributed:
        source_cnn = torch.nn.parallel.DistributedDataParallel(source_cnn, device_ids=[args.local_rank])
        target_cnn = torch.nn.parallel.DistributedDataParallel(target_cnn, device_ids=[args.local_rank])
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.local_rank])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(target_cnn.encoder, cfg, cfg.learning_rate)
    optimizer_discriminator = get_optimizer(discriminator, cfg, cfg.learning_rate_disc)

    wandb.watch((target_cnn.encoder, discriminator), log="all")

    if cfg.pretrained is not None:
        dist_print("Using pretrained model from ", cfg.pretrained)
        # Initialize target encoder with pretrained source encoder
        trained_model_state_dict = torch.load(cfg.pretrained)
        trained_model_state_dict = {(key.replace("module.", "") if "module." in key else key): value for key, value in trained_model_state_dict.items()}

        model_state_dict = {key: value for key, value in trained_model_state_dict["encoder"].items() if "model" in key or "aux" in key or "pool" in key}
        source_cnn.encoder.load_state_dict(model_state_dict, strict=False)
        target_cnn.encoder.load_state_dict(model_state_dict, strict=False)

        # Load classifier to validate and test accuracy
        cls_state_dict = {key: value for key, value in trained_model_state_dict["classifier"].items() if "cls" in key}
        source_cnn.classifier.load_state_dict(cls_state_dict, strict=False)
        target_cnn.classifier.load_state_dict(cls_state_dict, strict=False)

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k, v in state_all.items():
            if 'model' in k or 'aux' in k or 'pool' in k:
                state_clip[k] = v
        source_cnn.encoder.load_state_dict(state_clip, strict=False)
        target_cnn.encoder.load_state_dict(state_clip, strict=False)

    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)

        # Load encoder
        state_encoder = torch.load(cfg.resume)['encoder']
        source_cnn.encoder.load_state_dict(state_encoder, strict=False)
        target_cnn.encoder.load_state_dict(state_encoder, strict=False)

        # Load classifier
        state_classifier = torch.load(cfg.resume)['classifier']
        source_cnn.classifier.load_state_dict(state_classifier, strict=False)
        target_cnn.classifier.load_state_dict(state_classifier, strict=False)

        # Load discriminator
        state_discriminator = torch.load(cfg.resume)['discriminator']
        discriminator.load_state_dict(state_discriminator, strict=False)

        # Load optimizer
        if 'optimizer_enc' in state_all.keys():
            optimizer.load_state_dict(state_all['optimizer_enc'])
        if 'optimizer_disc' in state_all.keys():
            optimizer_discriminator.load_state_dict(state_all['optimizer_disc'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    for param in source_cnn.parameters():
        param.requires_grad = False

    for param in target_cnn.classifier.parameters():
        param.requires_grad = False

    scheduler = get_scheduler(optimizer, cfg, len(train_loader_source))
    scheduler_discriminator = get_scheduler(optimizer_discriminator, cfg, len(train_loader_source))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    cp_projects(args.auto_backup, work_dir)

    validate(source_cnn, val_loader_source, cfg.use_aux, 'sim_val')
    validate(target_cnn, val_loader_target, cfg.use_aux, 'real_val')
    validate_discriminator(val_loader_source, val_loader_target, cfg.use_aux)
    test(test_loader_target, cfg.use_aux)

    for epoch in range(resume_epoch, cfg.epoch):
        train(epoch, cfg.use_aux)
        validate(source_cnn, val_loader_source, cfg.use_aux, 'sim_val')
        validate(target_cnn, val_loader_target, cfg.use_aux, 'real_val')
        validate_discriminator(val_loader_source, val_loader_target, cfg.use_aux)
        test(test_loader_target, cfg.use_aux)

        if (epoch + 1) % 5 == 0:
            save_model(target_cnn, discriminator, optimizer, optimizer_discriminator, epoch, work_dir)
