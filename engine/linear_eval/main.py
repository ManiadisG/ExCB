import sys
import os

sys.path.append(os.getcwd())

import random
import numpy as np
import torch
import torch.distributed as dist
from engine.linear_eval.trainer import LinearEvalTrainer, build_linear_classifier
from utils.logger import Logger, update_wandb_run
from utils.arguments import load_arguments
from utils.ddp_utils import batch_size_per_device, init_distributed_mode
from data import build_dataset
from torch.utils.data import DataLoader, DistributedSampler
from engine.optimizer import BuildOptimizer
import traceback
import resource

def main(args, logger):
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logger.print("Loading data")
    dataset_train = build_dataset(args.dataset, "linear_train",args.dataset_path, args, logger)
    sampler_train = DistributedSampler(dataset_train)
    train_dataloader = DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size_per_device, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    dataset_val = build_dataset(args.dataset, "linear_eval", args.dataset_path,args, logger)
    sampler_val = DistributedSampler(dataset_val,shuffle=False)
    val_dataloader = DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size_per_device, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    logger.print("Loading model")
    model, criterion, checkpoint_args = build_linear_classifier(args, logger)
    model.to(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    optimizer, scheduler, lr_steps = BuildOptimizer.build_lc_optimizer(model, args, args.epochs)
    args.lr_exploration = lr_steps

    trainer = LinearEvalTrainer(model, criterion, optimizer, scheduler, args, logger)
    
    max_acc = 0
    for ep in range(args.epochs):
        acc = trainer.train_epoch(train_dataloader, val_dataloader)
        max_acc = max(acc, max_acc)
        
    if hasattr(checkpoint_args, "run_path") and args.rank==0 and args.wandb_mode=="online" and args.update_run:
        try:
            update_wandb_run(checkpoint_args.run_path, {"linear_eval_acc": acc})
        except:
            logger.print("Failed to upload accuracy to pretraining wandb job")

if __name__ == '__main__':
    print("Launching")
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    args = load_arguments("./engine/linear_eval/configs")
    print("Loaded arguments")
    init_distributed_mode(args)

    args.batch_size_per_device = batch_size_per_device(args.batch_size)
    logger = Logger(args)

    try:
        main(args, logger)
        logger.finish()
    except Exception as e:
        msg = traceback.format_exc()
        r = os.environ["RANK"]
        f = open(f"{args.output_dir}/rank_{int(r)}_error_log.txt", "a")
        f.write(str(msg))
        f.close()
        logger.print(msg)
        logger.finish(crashed=True)
        