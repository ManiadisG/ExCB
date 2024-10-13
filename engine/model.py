import torch
import torch.nn as nn
import torch.nn.functional as F
import architectures.backbones as backbones
from architectures.layers import MLPBottleneckClassifier, MLP
from utils.misc import export_fn
from utils.misc import load_file


@export_fn
def build_model(args, logger=None, distributed=True, checkpoint=None, teacher=False):
    model = Classifier(args, teacher=teacher)
    model.to(args.gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if logger is not None:
        p_count, p_count_train = 0, 0
        for p in model.parameters():
            p_count+=p.numel()
            if p.requires_grad:
                p_count_train+=p.numel()
        logger.print(f"Model built. {p_count} parameters, {p_count_train} trainable parameters.")
    if checkpoint is not None:
        logger.print("Loading model checkpoint")
        model.load_state_dict(checkpoint, strict=True)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    return model

@export_fn
def load_checkpoint_models(path, output_dir, logger):
    checkpoint = load_file(path, output_dir, logger)
    return {"student": checkpoint["student"], "teacher": checkpoint["teacher"]}

class Classifier(nn.Module):
    def __init__(self, args, teacher=False):
        super(Classifier, self).__init__()
        self.args = args
        if args.mlp_activation=="relu":
            mlp_activation=nn.ReLU
        elif args.mlp_activation=="gelu":
            mlp_activation=nn.GELU
        self.backbone = backbones.__dict__[args.backbone](args, teacher)
        self.classifier = MLPBottleneckClassifier(self.backbone.output_shape, args.num_clusters, args.mlp_hidden_dim, 
                                                  args.mlp_hidden_layers, bottleneck_dim=args.bottleneck_dim, bn=args.mlp_bn,
                                                  activation=mlp_activation, predictor = self.args.predictor, predictor_grads=False,
                                                  local_crop_grads=args.local_crop_grads, predictor_activation=args.predictor_activation)
        
    def forward(self, x, get_predictor=False, return_projections=False):
        if not isinstance(x, (list, tuple)):
            x = [x]
        features = [self.backbone(x_) for x_ in x]
        similarities, projections = self.classifier(features, get_predictor)
        if return_projections:
            return similarities, projections
        else:
            return similarities

    def get_projections(self, x):
        features = self.backbone(x)
        similarities, projections = self.classifier(features, get_predictor=False)
        return projections

    def get_parameters(self):
        parameters = {}
        parameters["backbone"] = [param for name, param in self.backbone.named_parameters() if not (name.endswith(".bias") or len(param.shape) == 1) and param.requires_grad]
        parameters["backbone_noreg"] = [param for name, param in self.backbone.named_parameters() if (name.endswith(".bias") or len(param.shape) == 1) and param.requires_grad]
        parameters["projector"] = [param for name, param in self.classifier.named_parameters() if (not (name.endswith(".bias") or len(param.shape) == 1) and "centroid_layer" not in name) and param.requires_grad]
        parameters["projector_noreg"] = [param for name, param in self.classifier.named_parameters() if ((name.endswith(".bias") or len(param.shape) == 1) and "centroid_layer" not in name) and param.requires_grad]
        parameters["centroids"] = [param for name, param in self.classifier.named_parameters() if "centroid_layer" in name and param.requires_grad]
        return parameters

class EMAHandler:
    def __init__(self, student, teacher, resumed=False):
        if not resumed:
            teacher.load_state_dict(student.state_dict(), strict=False)
        for param_ema in teacher.parameters():
            param_ema.requires_grad = False

    def ema_step(self, student, teacher, ema_momentum):
        for param, param_ema in zip(student.named_parameters(), teacher.named_parameters()):
            param_ema[1].data = param_ema[1].data * ema_momentum + param[1].data * (1. - ema_momentum)