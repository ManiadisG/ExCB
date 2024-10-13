from utils.logger import Logger
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.distributed as dist
import os
import logging
import torch.nn as nn
from utils.misc import export_fn, load_file, resnet_state_dict_check
from engine.model import build_model
from utils.arguments import config_dict_from_yaml
import numpy as np

logging.getLogger('matplotlib.font_manager').disabled = True

@export_fn
def build_linear_classifier(args, logger=None):
    args_dict = config_dict_from_yaml("./configs")
    checkpoint, checkpoint_args = load_checkpoint(args.path_to_model, args.model_version)
    checkpoint_args.gpu = args.gpu
    for k, v in args_dict.items():
        if not hasattr(checkpoint_args, k):
            setattr(checkpoint_args, k, v)    
    ssrl_model = build_model(checkpoint_args, distributed=False, teacher=True)
    model = LinearClassifier(ssrl_model.backbone, args)
    r = model.backbone.load_state_dict(checkpoint,strict=False)
    logger.print(f"Loaded checkpoint from {args.path_to_model}")
    logger.print(f"Checkpoint inconsistencies:")
    for r_ in r:
        logger.print(r_)
    criterion = nn.CrossEntropyLoss()
    if logger is not None:
        p_count, p_count_train = 0, 0
        for p in model.parameters():
            p_count+=p.numel()
            if p.requires_grad:
                p_count_train+=p.numel()
        logger.print(f"Model built. {p_count} parameters, {p_count_train} trainable parameters.")
    
    return model, criterion, checkpoint_args


class LinearClassifier(nn.Module):
    def __init__(self, backbone, args):
        super(LinearClassifier, self).__init__()
        self.backbone = backbone
        self.backbone.set_eval_state()

        lr_steps = np.linspace(args.lr-args.lr_exploration//2*args.lr_interval, args.lr+args.lr_exploration//2*args.lr_interval, args.lr_exploration+1)
        lr_steps = lr_steps[lr_steps>0]
        self.lr_exploration = len(lr_steps)
        
        for p in self.backbone.parameters():
            p.requires_grad=False
        module_list = [nn.Linear(self.backbone.output_shape, args.num_classes) for c in range(self.lr_exploration)]
        for m in module_list:
            m.weight.data.normal_(mean=0, std=0.01)
            m.bias.data.zero_()
        self.classifier_head = nn.ModuleList(module_list)
        for cl in self.classifier_head[1:]:
            cl.weight.data = self.classifier_head[0].weight.data
            cl.bias.data = self.classifier_head[0].bias.data
        
        

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = [classifier_head(x) for classifier_head in self.classifier_head]
        return x

def load_checkpoint(checkpoint_path, model_version="teacher"):
    # Customize to load model according to checkpoint structure
    # Optionally load pretraining wandb run path to update with final score
    map_location = "cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count())
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    checkpoint_args = checkpoint["args"]
    model_checkpoint = checkpoint[model_version]
    model_checkpoint_ = {}
    for k,v in model_checkpoint.items():
        if "backbone" in k:
            k = k.replace("backbone.","")
            model_checkpoint_[k]=v
    if "resnet" in checkpoint_args.backbone:
        model_checkpoint = resnet_state_dict_check(model_checkpoint_)
    return model_checkpoint_, checkpoint_args

@export_fn
class LinearEvalTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, args, logger:Logger):
        self.train_step=0
        self.epoch=0
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.logger = logger

        self.scaler = GradScaler()
        self.mixed_precision = args.__dict__.get("mixed_precision", True)
        self.logger.print(f"Mixed precision: {'ON' if self.mixed_precision else 'OFF'}")

    def save_checkpoint(self, path=None):
        if self.args.rank==0:
            state_dict = {"model": self.model.module.state_dict(),
                          "criterion":self.criterion.state_dict(),
                          "optimizer":self.optimizer.state_dict(),
                          "scheduler": self.scheduler.state_dict(),
                          "args":self.args,
                          "train_step":self.train_step,
                          "epoch":self.epoch}
            if path is None:
                path = f"{self.args.output_dir}/linear_eval_checkpoint.pth"
            torch.save(state_dict, path)

    def load_checkpoint(self, path, strict=True):
        checkpoint = load_file(path, self.args.output_dir, self.logger)
        self.load_state_dict(self.model.module, checkpoint["model"], "model", strict)
        self.load_state_dict(self.criterion, checkpoint["criterion"], "criterion", strict)
        self.load_state_dict(self.optimizer, checkpoint["optimizer"], "optimizer", strict)
        self.load_state_dict(self.scheduler, checkpoint["scheduler"], "scheduler", strict)
        self.args = checkpoint["args"]
        self.train_step = checkpoint["train_step"]
        self.epoch = checkpoint["epoch"]
        self.logger.print(f"Checkpoint loaded from {path}")

    def train_epoch(self, train_dataloader, val_dataloader, print_interval=None):
        self.model.eval()
        if print_interval is None:
            print_interval = max(len(train_dataloader)//20,5)
        lr, wd = self.scheduler.step(False)
        device = self.args.gpu
        epoch_steps = len(train_dataloader)
        train_dataloader.sampler.set_epoch(self.epoch)
        for batch_id, batch in enumerate(train_dataloader):
            self.optimizer.zero_grad()

            idx, samples, annotations = batch
            samples = samples.to(device,non_blocking=True)
            annotations = annotations.cuda(device,non_blocking=True)

            preds = self.model(samples)
            if not isinstance(preds, list):
                preds = [preds]
            losses = [self.criterion(pred, annotations) for pred in preds]

            sum(losses).backward()
            self.optimizer.step()

            with torch.no_grad():
                accs = [(pred.argmax(-1)==annotations).float().mean() for pred in preds]

            log_dict = {}
            for loss, acc, lr_st in zip(losses, accs, self.args.lr_exploration):
                log_dict[f"train_loss_{lr_st:.4f}"]=loss
                log_dict[f"train_acc_{lr_st:.4f}"]=acc
                if lr_st == self.args.lr:
                    self.logger.log({"train_loss":loss, "lr":lr, "train_acc":acc, "wd": wd})
            self.logger.log(log_dict)

            if print_interval>0 and batch_id%print_interval==0:
                self.logger.print_epoch_progress(batch_id, epoch_steps, self.epoch, self.args.epochs)

        eval_print_interval = max(len(val_dataloader)//10,5)
        eval_steps = len(val_dataloader)
        eval_accs = [torch.zeros((1,),device=self.args.gpu) for i in range(len(self.args.lr_exploration))]
        eval_accs5 = [torch.zeros((1,),device=self.args.gpu) for i in range(len(self.args.lr_exploration))]
        eval_samples = torch.zeros((1,),device=self.args.gpu)
        with torch.no_grad():
            for batch_id, batch in enumerate(val_dataloader):
                idx, samples, annotations = batch
                samples = samples.to(device,non_blocking=True)
                annotations = annotations.cuda(device,non_blocking=True)

                preds = self.model(samples)
                for i, pred in enumerate(preds):
                    pl = pred.argsort(dim=-1,descending=True)
                    eval_accs[i] += (pl[:, 0]==annotations).float().sum()
                    eval_accs5[i] += (pl[:, :5]==annotations.unsqueeze(-1)).max(-1)[0].float().sum()
                eval_samples+=len(samples)

                if batch_id%eval_print_interval==0:
                    self.logger.print(f"Validation step {batch_id} of {eval_steps}")
        
        dist.all_reduce(eval_samples)

        log_dict = {}

        [dist.all_reduce(eval_acc) for eval_acc in eval_accs]
        eval_accs = [eval_acc/eval_samples for eval_acc in eval_accs]
        log_dict["eval_acc"]=eval_accs[len(eval_accs)//2]
        log_dict["max_eval_acc"]=max(eval_accs)
        for acc, lr in zip(eval_accs, self.args.lr_exploration):
            log_dict[f"eval_acc_{lr:.4f}"]=acc

        [dist.all_reduce(eval_acc5) for eval_acc5 in eval_accs5]
        eval_accs5 = [eval_acc5/eval_samples for eval_acc5 in eval_accs5]
        log_dict["max_eval_acc5"]=max(eval_accs5)
        log_dict["eval_acc5"]=eval_accs5[len(eval_accs5)//2]
        for acc, lr in zip(eval_accs5, self.args.lr_exploration):
            log_dict[f"eval_acc5_{lr:.4f}"]=acc
        self.logger.log(log_dict)

        self.logger.epoch_end(self.epoch, self.args.epochs)
        self.epoch+=1
        lr, wd = self.scheduler.step(True)
        return float(log_dict["eval_acc"])

