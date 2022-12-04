import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn 
from PIL import Image
from apex import amp
from apex.parallel import DistributedDataParallel
from torch import distributed
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

import argparser
import tasks
import utils
from dataset import (AdeSegmentationIncremental,
                     CityscapesSegmentationIncrementalDomain,
                     VOCSegmentationIncremental, transform)
from metrics import StreamSegMetrics
from segmentation_module import make_model
from train import Trainer
from utils.logger import Logger


def save_ckpt(path, model, trainer, optimizer, scheduler, epoch, best_score):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "trainer_state": trainer.state_dict()
    }
    torch.save(state, path)


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = transform.Compose(
        [
            transform.RandomResizedCrop(opts.crop_size, (0.5, 2.0)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if opts.crop_val:
        val_transform = transform.Compose(
            [
                transform.Resize(size=opts.crop_size),
                transform.CenterCrop(size=opts.crop_size),
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        # no crop, batch size = 1
        val_transform = transform.Compose(
            [
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)
    labels_cum = labels_old + labels

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    elif opts.dataset == 'ade':
        dataset = AdeSegmentationIncremental
    elif opts.dataset == 'cityscapes_domain':
        dataset = CityscapesSegmentationIncrementalDomain
    else:
        raise NotImplementedError

    if opts.overlap:
        path_base += "-ov"

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)

    train_dst = dataset(
        root=opts.data_root,
        train=True,
        transform=train_transform,
        labels=list(labels),
        labels_old=list(labels_old),
        idxs_path=path_base + f"/train-{opts.step}.npy",
        masking=not opts.no_mask,
        overlap=opts.overlap,
        disable_background=opts.disable_background,
        data_masking=opts.data_masking,
        test_on_val=opts.test_on_val,
        step=opts.step
    )

    if not opts.no_cross_val:  # if opts.cross_val:
        train_len = int(0.8 * len(train_dst))
        val_len = len(train_dst) - train_len
        train_dst, val_dst = torch.utils.data.random_split(train_dst, [train_len, val_len])
    else:  # don't use cross_val
        val_dst = dataset(
            root=opts.data_root,
            train=False,
            transform=val_transform,
            labels=list(labels),
            labels_old=list(labels_old),
            idxs_path=path_base + f"/val-{opts.step}.npy",
            masking=not opts.no_mask,
            overlap=True,
            disable_background=opts.disable_background,
            data_masking=opts.data_masking,
            step=opts.step
        )

    image_set = 'train' if opts.val_on_trainset else 'val'
    test_dst = dataset(
        root=opts.data_root,
        train=opts.val_on_trainset,
        transform=val_transform,
        labels=list(labels_cum),
        idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy",
        disable_background=opts.disable_background,
        test_on_val=opts.test_on_val,
        step=opts.step,
        ignore_test_bg=opts.ignore_test_bg
    )

    return train_dst, val_dst, test_dst, len(labels_cum)


def main(opts):
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = opts.local_rank, torch.device(opts.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    if len(opts.lr) == 1 and len(opts.step) > 1:
        opts.lr = opts.lr * len(opts.step)

    os.makedirs("results", exist_ok=True)

    print(f"Learning for {len(opts.step)} with lrs={opts.lr}.")
    all_val_scores = []
    for i, (step, lr) in enumerate(zip(copy.deepcopy(opts.step), copy.deepcopy(opts.lr))):
        if i > 0:
            opts.step_ckpt = None

        opts.step = step
        opts.lr = lr

        val_score = run_step(opts, world_size, rank, device)
        if rank == 0:
            all_val_scores.append(val_score)

        torch.cuda.empty_cache()

        if rank == 0:
            with open(f"results/{opts.date}_{opts.dataset}_{opts.task}_{opts.name}.csv", "a+") as f:
                classes_iou = ','.join(
                    [str(val_score['Class IoU'].get(c, 'x')) for c in range(opts.num_classes)]
                )
                f.write(f"{step},{classes_iou},{val_score['Mean IoU']}\n")


def run_step(opts, world_size, rank, device):
    # Initialize logging
    task_name = f"{opts.task}-{opts.dataset}"
    logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    if rank == 0:
        logger = Logger(
            logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step
        )
    else:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=False)

    logger.print(f"Device: {device}")

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # xxx Set up dataloader
    train_dst, val_dst, test_dst, n_classes = get_dataset(opts)
    # reset the seed, this revert changes in random seed
    random.seed(opts.random_seed)

    train_loader = data.DataLoader(
        train_dst,
        batch_size=opts.batch_size,
        sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
        num_workers=opts.num_workers,
        drop_last=True
    )
    val_loader = data.DataLoader(
        val_dst,
        batch_size=opts.batch_size if opts.crop_val else 1,
        sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
        num_workers=opts.num_workers
    )
    logger.info(
        f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
        f" Test set: {len(test_dst)}, n_classes {n_classes}"
    )
    logger.info(f"Total batch size is {opts.batch_size * world_size}")

    # xxx Set up model
    logger.info(f"Backbone: {opts.backbone}")

    opts.inital_nb_classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)[0]

    step_checkpoint = None
    model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
    logger.info(f"[!] Model made with{'out' if opts.no_pretrained else ''} pre-trained")

    if opts.step == 0:  # if step 0, we don't need to instance the model_old
        model_old = None
    else:  # instance model_old
        model_old = make_model(
            opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1)
        )

    if opts.fix_bn:
        model.fix_bn()

    logger.debug(model)

    # xxx Set up optimizer
    params = []
    if not opts.freeze:
        params.append(
            {
                "params": filter(lambda p: p.requires_grad, model.body.parameters()),
                'weight_decay': opts.weight_decay
            }
        )

    params.append(
        {
            "params": filter(lambda p: p.requires_grad, model.head.parameters()),
            'weight_decay': opts.weight_decay
        }
    )

    if opts.lr_old is not None and opts.step > 0:
        params.append(
            {
                "params": filter(lambda p: p.requires_grad, model.cls[:-1].parameters()),
                'weight_decay': opts.weight_decay,
                "lr": opts.lr_old * opts.lr
            }
        )
        params.append(
            {
                "params": filter(lambda p: p.requires_grad, model.cls[-1:].parameters()),
                'weight_decay': opts.weight_decay
            }
        )
    else:
        params.append(
            {
                "params": filter(lambda p: p.requires_grad, model.cls.parameters()),
                'weight_decay': opts.weight_decay
            }
        )
    if model.scalar is not None:
        params.append({"params": model.scalar, 'weight_decay': opts.weight_decay})

    optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(
            optimizer, max_iters=opts.epochs * len(train_loader), power=opts.lr_power
        )
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor
        )
    else:
        raise NotImplementedError
    logger.debug("Optimizer:\n%s" % optimizer)

    if model_old is not None:
        [model, model_old], optimizer = amp.initialize(
            [model.to(device), model_old.to(device)], optimizer, opt_level=opts.opt_level
        )
        model_old = DistributedDataParallel(model_old)
    else:
        model, optimizer = amp.initialize(model.to(device), optimizer, opt_level=opts.opt_level)

    # Put the model on GPU
    model = DistributedDataParallel(model, delay_allreduce=True)

    # xxx Load old model from old weights if step > 0!
    if opts.step > 0:
        # get model path
        if opts.step_ckpt is not None:
            path = opts.step_ckpt
        else:
            path = f"{opts.checkpoint}/{task_name}_{opts.name}_{opts.step - 1}.pth"

        # generate model from path
        if os.path.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")
            model.load_state_dict(
                step_checkpoint['model_state'], strict=False
            )  # False because of incr. classifiers
            if opts.init_balanced:
                # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
                model.module.init_new_classifier(device)
            elif opts.init_multimodal is not None:
                model.module.init_new_classifier_multimodal(
                    device, train_loader, opts.init_multimodal
                )
            # Load state dict from the model state dict, that contains the old model parameters
            model_old.load_state_dict(
                step_checkpoint['model_state'], strict=opts.strict_weights
            )  # Load also here old parameters
            logger.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint['model_state']
        elif opts.debug:
            logger.info(
                f"[!] WARNING: Unable to find of step {opts.step - 1}! Do you really want to do from scratch?"
            )
        else:
            raise FileNotFoundError(path)
        # put the old model into distributed memory and freeze it
        for par in model_old.parameters():
            par.requires_grad = False
        model_old.eval()

    # xxx Set up Trainer
    trainer_state = None
    # if not first step, then instance trainer from step_checkpoint
    if opts.step > 0 and step_checkpoint is not None:
        if 'trainer_state' in step_checkpoint:
            trainer_state = step_checkpoint['trainer_state']

    # instance trainer (model must have already the previous step weights)
    trainer = Trainer(
        model,
        model_old,
        device=device,
        opts=opts,
        trainer_state=trainer_state,
        classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step),
        step=opts.step
    )

    # xxx Handle checkpoint for current model (model old will always be as previous step or None)
    best_score = 0.0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=opts.strict_weights)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint['best_score']
        logger.info("[!] Model restored from %s" % opts.ckpt)
        # if we want to resume training, resume trainer from checkpoint
        if 'trainer_state' in checkpoint:
            trainer.load_state_dict(checkpoint['trainer_state'])
        del checkpoint
    else:
        if opts.step == 0:
            logger.info("[!] Train from scratch")

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    if rank == 0 and opts.sample_num > 0:
        sample_ids = np.random.choice(
            len(val_loader), opts.sample_num, replace=False
        )  # sample idxs for visualization
        logger.info(f"The samples id are {sample_ids}")
    else:
        sample_ids = None

    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )  # de-normalization for original images

    TRAIN = not opts.test
    if opts.dataset == "cityscapes_domain":
        val_metrics = StreamSegMetrics(opts.num_classes)
    else:
        val_metrics = StreamSegMetrics(n_classes)
    results = {}

    # check if random is equal here.
    logger.print(torch.randint(0, 100, (1, 1)))
    # train/val here

    if TRAIN:
        trainer.before(train_loader=train_loader, logger=logger)

        for cur_epoch in range(opts.epochs):
            # =====  Train  =====
            model.train()

            epoch_loss = trainer.train(
                cur_epoch=cur_epoch,
                optim=optimizer,
                train_loader=train_loader,
                scheduler=scheduler,
                logger=logger
            )

            logger.info(
                f"End of Epoch {cur_epoch}/{opts.epochs}, Average Loss={epoch_loss[0]+epoch_loss[1]},"
                f" Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]}"
            )

            # =====  Log metrics on Tensorboard =====
            logger.add_scalar("E-Loss", epoch_loss[0] + epoch_loss[1], cur_epoch)
            logger.add_scalar("E-Loss-reg", epoch_loss[1], cur_epoch)
            logger.add_scalar("E-Loss-cls", epoch_loss[0], cur_epoch)

            # =====  Validation  =====
            if (cur_epoch + 1) % opts.val_interval == 0:
                logger.info("validate on val set...")
                model.eval()
                val_loss, val_score, ret_samples = trainer.validate(
                    loader=val_loader,
                    metrics=val_metrics,
                    ret_samples_ids=sample_ids,
                    logger=logger
                )

                logger.print("Done validation")
                logger.info(
                    f"End of Validation {cur_epoch}/{opts.epochs}, Validation Loss={val_loss[0]+val_loss[1]},"
                    f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}"
                )

                logger.info(val_metrics.to_str(val_score))

                # =====  Save Best Model  =====
                if rank == 0:  # save best model at the last iteration
                    score = val_score['Mean IoU']
                    # best model to build incremental steps
                    save_ckpt(
                        f"{opts.checkpoint}/{task_name}_{opts.name}_{opts.step}.pth", model,
                        trainer, optimizer, scheduler, cur_epoch, score
                    )
                    logger.info("[!] Checkpoint saved.")

                # =====  Log metrics on Tensorboard =====
                # visualize validation score and samples
                logger.add_scalar("V-Loss", val_loss[0] + val_loss[1], cur_epoch)
                logger.add_scalar("V-Loss-reg", val_loss[1], cur_epoch)
                logger.add_scalar("V-Loss-cls", val_loss[0], cur_epoch)
                logger.add_scalar("Val_Overall_Acc", val_score['Overall Acc'], cur_epoch)
                logger.add_scalar("Val_MeanIoU", val_score['Mean IoU'], cur_epoch)
                logger.add_table("Val_Class_IoU", val_score['Class IoU'], cur_epoch)
                logger.add_table("Val_Acc_IoU", val_score['Class Acc'], cur_epoch)
                # logger.add_figure("Val_Confusion_Matrix", val_score['Confusion Matrix'], cur_epoch)

                # keep the metric to print them at the end of training
                results["V-IoU"] = val_score['Class IoU']
                results["V-Acc"] = val_score['Class Acc']

                for k, (img, target, lbl) in enumerate(ret_samples):
                    img = (denorm(img) * 255).astype(np.uint8)
                    target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
                    lbl = label2color(lbl).transpose(2, 0, 1).astype(np.uint8)

                    concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                    logger.add_image(f'Sample_{k}', concat_img, cur_epoch)

    # =====  Save Best Model at the end of training =====
    if rank == 0 and TRAIN:  # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(
            f"{opts.checkpoint}/{task_name}_{opts.name}_{opts.step}.pth", model, trainer, optimizer,
            scheduler, cur_epoch, best_score
        )
        logger.info("[!] Checkpoint saved.")

    torch.distributed.barrier()

    # xxx From here starts the test code
    logger.info("*** Test the model on all seen classes...")
    # make data loader
    test_loader = data.DataLoader(
        test_dst,
        batch_size=opts.batch_size if opts.crop_val else 1,
        sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
        num_workers=opts.num_workers
    )

    # load best model
    if True: #TRAIN:
        # Always reloading model for now
        # https://github.com/arthurdouillard/CVPR2021_PLOP/issues/3
        model = make_model(
            opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)
        )
        # Put the model on GPU
        model = DistributedDataParallel(model.cuda(device))
        ckpt = f"{opts.checkpoint}/{task_name}_{opts.name}_{opts.step}.pth"
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        logger.info(f"*** Model restored from {ckpt}")
        del checkpoint
        trainer = Trainer(model, None, device=device, opts=opts, step=opts.step)

    model.eval()

    val_loss, val_score, _ = trainer.validate(
        loader=test_loader, metrics=val_metrics, logger=logger, end_task=True
    )
    logger.print("Done test")
    logger.info(
        f"*** End of Test, Total Loss={val_loss[0]+val_loss[1]},"
        f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}"
    )
    logger.info(val_metrics.to_str(val_score))
    logger.add_table("Test_Class_IoU", val_score['Class IoU'])
    logger.add_table("Test_Class_Acc", val_score['Class Acc'])
    logger.add_figure("Test_Confusion_Matrix", val_score['Confusion Matrix'])
    results["T-IoU"] = val_score['Class IoU']
    results["T-Acc"] = val_score['Class Acc']
    logger.add_results(results)

    logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'], opts.step)
    logger.add_scalar("T_MeanIoU", val_score['Mean IoU'], opts.step)
    logger.add_scalar("T_MeanAcc", val_score['Mean Acc'], opts.step)

    logger.close()

    del model
    if model_old is not None:
        del model_old

    return val_score

####### merged branches ####### begin xjw fixed
def merge(conv, bias, bn, bn_bias, running_means, running_vars):
    """ 请参考mergex函数，对merge函数进行补全
    """

def mergex(conv, bias, bn, bn_bias, running_means, running_vars, pos):
    # if bias is not None:
    #     conv_bias = conv_bias.clone().to(conv2d.weight.device)    
    k = conv.clone()
    running_mean = running_means[pos*256:(1+pos)*256]
    running_var = running_vars[pos*256:(1+pos)*256]
    eps = 1e-5
    gamma = bn.abs()[pos*256:(1+pos)*256] + eps
    beta = bn_bias[pos*256:(1+pos)*256]

    gamma = gamma / 2.
    beta = beta / 2.

    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    if bias is not None:
        return k * t, beta - running_mean * gamma / std + t.view(-1) * bias.view(-1)
    else:
        return k * t, beta - running_mean * gamma / std

def init_right(conv2d, bn2d, conv2d_new, bn2d_new, init_type):
    if init_type == 'original':
        return conv2d_new, bn2d_new
    return conv2d_new, bn2d_new

def convert_model(checkpoint, load_dict=None):
    new_ckpt = {}
    for name, mm in checkpoint["model_state"].items():
        if 'convs.conv2.weight' in name:
            conv2 = mm
            conv2_bias = checkpoint["model_state"][name.replace("weight", "bias")]
            conv2_new = checkpoint["model_state"][name.replace("conv2", "conv2_new")]
            bn2 = checkpoint["model_state"][name.replace("conv2", "bn2")]
            bn2_bias = checkpoint["model_state"][name.replace("conv2.weight", "bn2.bias")]
            bn2_runningmean = checkpoint["model_state"][name.replace("conv2.weight", "bn2.running_mean")]
            bn2_runningvar = checkpoint["model_state"][name.replace("conv2.weight", "bn2.running_var")]
            bn2_new = checkpoint["model_state"][name.replace("conv2.weight", "bn2_new.weight")]
            bn2_new_bias = checkpoint["model_state"][name.replace("conv2.weight", "bn2_new.bias")]
            bn2_new_runningmean = checkpoint["model_state"][name.replace("conv2.weight", "bn2_new.running_mean")]
            bn2_new_runningvar = checkpoint["model_state"][name.replace("conv2.weight", "bn2_new.running_var")]

            k1, b1 = merge(conv2, conv2_bias, bn2, bn2_bias, bn2_runningmean, bn2_runningvar)
            k2, b2 = merge(conv2_new, None, bn2_new, bn2_new_bias, bn2_new_runningmean, bn2_new_runningvar)
            k = k1 + k2
            b = b1 + b2
            new_ckpt[name] = k.clone()
            new_ckpt[name.replace("weight", "bias")] = b 
            new_ckpt[name.replace("conv2", "bn2")] = torch.ones((bn2.shape[0],))[:]
            new_ckpt[name.replace("conv2.weight", "bn2.bias")] = torch.zeros((bn2.shape[0],))[:]
        elif 'map_convs.0.weight' in name:
            for i in range(4):
                conv = checkpoint["model_state"][name.replace("0.weight", "{}.weight".format(i))]
                conv_bias = checkpoint["model_state"][name.replace("0.weight", "{}.bias".format(i))]
                conv_new = checkpoint["model_state"][name.replace("convs.0.weight", "convs_new.{}.weight".format(i))]
                bn = checkpoint["model_state"][name.replace("convs.0.weight", "bn.weight")]
                bn_bias = checkpoint["model_state"][name.replace("convs.0.weight", "bn.bias")]
                bn_runningmean = checkpoint["model_state"][name.replace("convs.0.weight", "bn.running_mean")]
                bn_runningvar = checkpoint["model_state"][name.replace("convs.0.weight", "bn.running_var")]
                bn_new = checkpoint["model_state"][name.replace("convs.0.weight", "bn_new.weight")]
                bn_new_bias = checkpoint["model_state"][name.replace("convs.0.weight", "bn_new.bias")]
                bn_new_runningmean = checkpoint["model_state"][name.replace("convs.0.weight", "bn_new.running_mean")]
                bn_new_runningvar = checkpoint["model_state"][name.replace("convs.0.weight", "bn_new.running_var")]
                k1, b1 = mergex(conv, conv_bias, bn, bn_bias, bn_runningmean, bn_runningvar, i)
                k2, b2 = mergex(conv_new, None, bn_new, bn_new_bias, bn_new_runningmean, bn_new_runningvar, i)
                k = k1 + k2
                b = b1 + b2
                new_ckpt[name.replace("0", "{}".format(i))] = k
                new_ckpt[name.replace("0.weight", "{}.bias".format(i))] = b 
            new_ckpt[name.replace("convs.0.weight", "bn.weight")] = torch.ones((mm.shape[0], ))[:]
            new_ckpt[name.replace("convs.0.weight", "bn.bias")] = torch.zeros((mm.shape[0], ))[:]
        elif 'convs.conv2' not in name and 'convs.bn2' not in name and 'map_convs' not in name:
            new_ckpt[name] = mm
    return new_ckpt
####### merged branches ####### end
#################################################################

def val(opts):
    
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = opts.local_rank, torch.device(opts.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    os.makedirs("results", exist_ok=True)
    
    opts.step = opts.step[0]
    task_name = f"{opts.task}-{opts.dataset}"


    val_transform = transform.Compose(
            [
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    pic = Image.open("./2007_000452.jpg").convert("RGB")
    pic = val_transform(pic).cuda()
    pic = pic.unsqueeze(0)


    model = make_model(
            opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)
        )
        # Put the model on GPU
    model = DistributedDataParallel(model.cuda(device))

    model_old = make_model(
            opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, 1)
        )
    model_old = DistributedDataParallel(model_old.cuda(device))

    trainer = Trainer(model, None, device=device, opts=opts, step=opts.step)
    trainer_old = Trainer(model_old, None, device=device, opts=opts, step=opts.step)


    ckpt = f"./test.pth"
    checkpoint = torch.load(ckpt, map_location="cpu")

    ckpt_old = f"./compare.pth"
    checkpoint_old = torch.load(ckpt_old, map_location="cpu")
    # for name, p in checkpoint['model_state'].items():
    #     print(name)
    new_ckpt = convert_model(checkpoint)
    new_ckpt_old = convert_model(checkpoint_old)
    for name, mm in model.named_modules():
        if hasattr(mm, 'convs'):
            mm.convs.conv2.bias = nn.Parameter(torch.zeros(mm.convs.conv2.weight.shape[0]).to(mm.convs.conv2.weight.device))
            # print(mm.convs.bn2.eps)
            mm.convs.bn2.eps = 0
            mm.convs.bn2.running_mean.data[:] = torch.zeros((mm.convs.bn2.weight.shape[0],))[:]
            mm.convs.bn2.running_var.data[:] = torch.ones((mm.convs.bn2.weight.shape[0],))[:]
        if hasattr(mm, 'map_convs'):
            for kk in range(4):
                mm.map_convs[kk].bias = nn.Parameter(torch.zeros(mm.map_convs[kk].weight.shape[0]).to(mm.map_convs[kk].weight.device))
            mm.map_bn.running_var.data[:] = torch.ones((mm.map_bn.weight.shape[0],))[:]
            mm.map_bn.eps = 0
            mm.map_bn.running_mean.data[:] = torch.zeros((mm.map_bn.weight.shape[0],))[:]
    
    for name, mm in model_old.named_modules():
        if hasattr(mm, 'convs'):
            mm.convs.conv2.bias = nn.Parameter(torch.zeros(mm.convs.conv2.weight.shape[0]).to(mm.convs.conv2.weight.device))
            # print(mm.convs.bn2.eps)
            mm.convs.bn2.eps = 0
            mm.convs.bn2.running_mean.data[:] = torch.zeros((mm.convs.bn2.weight.shape[0],))[:]
            mm.convs.bn2.running_var.data[:] = torch.ones((mm.convs.bn2.weight.shape[0],))[:]
        if hasattr(mm, 'map_convs'):
            for kk in range(4):
                mm.map_convs[kk].bias = nn.Parameter(torch.zeros(mm.map_convs[kk].weight.shape[0]).to(mm.map_convs[kk].weight.device))
            mm.map_bn.running_var.data[:] = torch.ones((mm.map_bn.weight.shape[0],))[:]
            mm.map_bn.eps = 0
            mm.map_bn.running_mean.data[:] = torch.zeros((mm.map_bn.weight.shape[0],))[:]

    model.load_state_dict(new_ckpt, strict=False)
    model_old.load_state_dict(new_ckpt_old, strict=False)
    model.eval()
    model_old.eval()

    
    import utils.utils

    label2color = utils.utils.Label2Color(cmap=utils.utils.color_map("voc"))

    output = model(pic)
    logits = output[0].max(dim=1)[1].cpu().numpy()
    pred_mask = label2color(logits)[0].astype(np.uint8)
    Image.fromarray(pred_mask).save(f"./results/step5_ours.png")

    output_old = model_old(pic)
    logits_old = output_old[0].max(dim=1)[1].cpu().numpy()
    pred_mask = label2color(logits_old)[0].astype(np.uint8)
    Image.fromarray(pred_mask).save(f"./results/step5_compare.png")

if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    os.makedirs(f"{opts.checkpoint}", exist_ok=True)

    # main(opts)
    val(opts)
