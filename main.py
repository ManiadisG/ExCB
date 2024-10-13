
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from engine import Trainer
from utils.logger import Logger
from utils.arguments import load_arguments
from utils.ddp_utils import batch_size_per_device, init_distributed_mode
from utils.misc import scheduling
from data import build_dataset
from torch.utils.data import DataLoader, DistributedSampler
from engine import build_model, BuildOptimizer, load_checkpoint_models
from engine.criterion import get_criterion
import traceback
import os
import resource
import wandb
import sys

def main(args, logger):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    logger.print("Loading data")

    dataset_train = build_dataset(args.dataset, "train", args.dataset_path, args, logger)
    sampler_train = DistributedSampler(dataset_train)
    train_dataloader = DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size_per_device, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    train_steps = len(train_dataloader)*args.epochs
    train_steps_per_epoch = len(train_dataloader)
    logger.total_steps = train_steps_per_epoch*args.epochs

    logger.print("Loading model")
    checkpoints = {}
    if args.checkpoint_path is not None:
        checkpoints = load_checkpoint_models(args.checkpoint_path, args.output_dir, logger)
    student = build_model(args, logger, checkpoint=checkpoints.get("student"))
    teacher = build_model(args, logger, checkpoint=checkpoints.get("teacher"), teacher=True)

    criterion = get_criterion(args, args.num_clusters, train_steps, train_steps_per_epoch, logger)

    warmup_epochs = args.__dict__.get("warmup_epochs",0)
    warmup_steps = len(train_dataloader)*warmup_epochs
    optimizer, scheduler = BuildOptimizer.build_optimizer(student, args, train_steps, warmup_steps)

    trainer = Trainer(student, teacher, criterion, optimizer, scheduler, args, logger)
    if args.checkpoint_path is not None:
        trainer.load_checkpoint(args.checkpoint_path)
        just_loaded=True
    else:
        just_loaded=False

    for ep in range(trainer.epoch, args.epochs):
        if not just_loaded:
            if ep%(args.epochs//10)==0:
                trainer.save_checkpoint(f"{trainer.args.output_dir}/checkpoint_ep{ep}.pth")
            if ep%2==0:
                trainer.save_checkpoint()
        just_loaded=False
        trainer.train_epoch(train_dataloader)
        trainer.logger.epoch_end(trainer.epoch, trainer.args.epochs)
        trainer.epoch+=1
    trainer.save_checkpoint()

if __name__ == '__main__':
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    args = load_arguments("./configs")
    init_distributed_mode(args)

    args.batch_size_per_device = batch_size_per_device(args.batch_size)
    logger = Logger(args)
    sys.stderr = sys.stdout
    args = logger.args

    try:
        main(args, logger)
        logger.print("\nJob finished\n")
        logger.finish()
    except Exception as e:
        msg = traceback.format_exc()
        r = os.environ["RANK"]
        f = open(f"{args.output_dir}/rank_{int(r)}_error_log.txt", "a")
        f.write(str(msg))
        f.close()
        logger.print(msg)
        logger.finish(crashed=True)