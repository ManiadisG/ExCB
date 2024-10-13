import math
from utils.misc import export_fn, scheduling
import torch
from torch.optim import Optimizer
import numpy as np

@export_fn
class BuildOptimizer:
    @staticmethod
    def build_optimizer(model, args, steps=None, warmup_steps=0):
        args_dict = args.__dict__
        parameters_dict = model.module.get_parameters()
        lr = args_dict.get("lr",0.0001)
        parameters = [{'params': parameters_dict["backbone"],"lr":lr, "weight_decay": args_dict.get("weight_decay",0.)},
                    {'params': parameters_dict["backbone_noreg"],"lr":lr, "weight_decay": 0.},
                    {'params': parameters_dict["projector"],"lr":lr, "weight_decay": args_dict.get("weight_decay",0.)},
                    {'params': parameters_dict["projector_noreg"],"lr":lr, "weight_decay": 0.},
                    {'params': parameters_dict["centroids"],"lr":lr, "weight_decay": args_dict.get("weight_decay",0.)},
                    ]
        optimizer = BuildOptimizer._set_optimizer(args, parameters)
        param_group_default = {"lr": [pg["lr"] for pg in parameters], "wd": [pg["weight_decay"] for pg in parameters]}
        scheduler = LRScheduler(optimizer, steps, warmup_steps, args, param_group_default)
        return optimizer, scheduler

    @staticmethod
    def build_lc_optimizer(model, args, steps=None, warmup_steps=0):
        args_dict = args.__dict__
        parameters = [p for p in model.parameters() if p.requires_grad]
        lr_steps = np.linspace(args.lr-args.lr_exploration//2*args.lr_interval, args.lr+args.lr_exploration//2*args.lr_interval, args.lr_exploration+1)
        lr_steps = lr_steps[lr_steps>0]
        parameters_list = []
        for i, lr in enumerate(lr_steps):
            parameters_list.append({'params': parameters[i*2], "lr": lr, "weight_decay": 0.})
            parameters_list.append({'params': parameters[i*2+1], "lr": lr, "weight_decay": 0.})
        optimizer = BuildOptimizer._set_optimizer(args, parameters_list)
        param_group_default = {"lr": [pg["lr"] for pg in parameters_list], "wd": [pg["weight_decay"] for pg in parameters_list]}
        scheduler = LRScheduler(optimizer, steps, warmup_steps, args, param_group_default)
        return optimizer, scheduler, lr_steps
    
    @staticmethod
    def build_semisup_optimizer(model, args, steps=None, warmup_steps=0):
        args_dict = args.__dict__
        parameters_backbone = [p for p in model.module.backbone.parameters() if p.requires_grad]
        parameters_head = [p for p in model.module.classifier_head.parameters() if p.requires_grad]
        parameters = [{'params': parameters_backbone,"lr":args.lr, "weight_decay": args_dict.get("weight_decay",0.)},
                      {'params': parameters_head,"lr":args.lr_head, "weight_decay": 0.}]
        optimizer = BuildOptimizer._set_optimizer(args, parameters)
        param_group_default = {"lr": [pg["lr"] for pg in parameters], "wd": [pg["weight_decay"] for pg in parameters]}
        scheduler = LRScheduler(optimizer, steps, warmup_steps, args, param_group_default)
        return optimizer, scheduler

    @staticmethod
    def _set_optimizer(args, parameters):
        args_dict = args.__dict__
        if args.optimizer.lower()=="adam":
            for p in parameters:
                p["betas"] = args_dict.get("betas",(0.9, 0.999))
            parameters = [p for p in parameters if len(p['params'])>0]
            optimizer = torch.optim.Adam(parameters)
        elif args.optimizer.lower()=="lars":
            for p in parameters:
                p["momentum"] = args_dict.get("sgd_momentum",0.9)
            parameters = [p for p in parameters if len(p['params'])>0]
            optimizer = LARS(parameters)
        elif args.optimizer.lower()=="sgd":
            for p in parameters:
                p["momentum"] = args_dict.get("sgd_momentum",0.9)
            parameters = [p for p in parameters if len(p['params'])>0]
            optimizer = torch.optim.SGD(parameters)
        elif args.optimizer.lower()=="adamw":
            for p in parameters:
                p["betas"] = args_dict.get("betas",(0.9, 0.999))
            parameters = [p for p in parameters if len(p['params'])>0]
            optimizer = torch.optim.AdamW(parameters)
        return optimizer

class LRScheduler:
    def __init__(self, optimizer:Optimizer, steps, warmup_steps, args, param_group_default = None):
        self.optimizer = optimizer
        self.schedule = args.__dict__.get("lr_schedule",None)
        self.steps = steps
        self.warmup_steps = warmup_steps
        self.lr_steps = args.__dict__.get("lr_steps",None)
        self.args = args
        self.current_step=0
        self.param_group_default = param_group_default

        self.weight_decay_target_factor = args.__dict__.get("weight_decay_target_factor", 1.)
        self.centroid_adjustments = args.__dict__.get("centroid_adjustments", None)

        # Attributes to be saved in checkpoints
        self.to_save = ["current_step"]

    def step(self, do_step=True):
        if do_step:
            self.current_step+=1
        for i, pg in enumerate(self.optimizer.param_groups):
            if self.param_group_default is None or self.param_group_default["lr"] is None:
                new_lr = self._get_lr()
            else:
                new_lr = self._get_lr(self.param_group_default["lr"][i])
            pg["lr"]=new_lr

            if self.weight_decay_target_factor==1.:
                new_wd = self.param_group_default["wd"][i]
            else:
                wd = self.param_group_default["wd"][i]
                wd_target = wd * self.weight_decay_target_factor
                new_wd = scheduling(max(wd_target, wd), min(wd_target, wd), self.current_step, self.steps,
                                    "cosine", wd_target<wd)
            if pg["weight_decay"]>0:
                pg["weight_decay"]=new_wd
            if (i+1)==len(self.optimizer.param_groups) and self.centroid_adjustments is not None: # modifications to the centroid layer
                if self.centroid_adjustments=="wdx10c1":
                    pg["weight_decay"] = new_wd*(scheduling(1, 0, self.current_step, self.steps, "cosine", True)*9+1)
        return self._get_lr(), new_wd


    def _get_lr(self, lr=None):
        if lr is None:
            lr = self.args.lr
        if self.schedule=="cosine":
            lr_max = lr
            lr_min_factor = self.args.__dict__.get("lr_min_factor",0.01)
            lr_min = lr * lr_min_factor
            if self.current_step<self.warmup_steps:
                lr = lr_max*self.current_step/self.warmup_steps
            else:
                lr= lr_min + 0.5 * (lr_max - lr_min) * \
                    (1 + math.cos(math.pi * (self.current_step-self.warmup_steps) / (self.steps-self.warmup_steps)))
        elif self.schedule=="step":
            for lr_step in self.lr_steps:
                if self.current_step>=lr_step:
                    lr = lr*self.args.__dict__.get("lr_gamma",0.1)
        return lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key in self.to_save}

    def load_state_dict(self, state_dict, strict=False):
        self.__dict__.update(state_dict)


class LARS(torch.optim.Optimizer):
    """
    From https://github.com/facebookresearch/dino
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms