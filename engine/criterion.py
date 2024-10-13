import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as distributed
import einops
import numpy as np
import matplotlib.pyplot as plt
from utils.ddp_utils import cat_all_gather
from sklearn.metrics.cluster import normalized_mutual_info_score

def get_criterion(args, num_clusters, train_steps, steps_per_epoch, logger):
    return ClusteringLoss(num_clusters, args.epochs, train_steps, steps_per_epoch, args.teacher_temp,
                            args.student_temp, args.center_momentum,
                            args, logger).to(args.gpu)

class ClusteringLoss(nn.Module):
    def __init__(self, out_dim, nepochs, nsteps, steps_per_epoch, teacher_temp=0.07,
                 student_temp=0.1, center_momentum=0.99,
                 args=None, logger=None):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.logger = logger
        self.nepochs = nepochs
        self.steps_per_epoch = steps_per_epoch
        self.nsteps = nsteps
        self.out_dim = out_dim
        self.args = args

        self.register_buffer("center", torch.ones(1, out_dim)/out_dim)
        self.register_buffer("cluster_count", torch.zeros((out_dim,)))

    def reset_centering(self):
        self.center = self.center*0+1/self.out_dim

    def loss_func(self, p, q):
        return torch.sum(- q * F.log_softmax(p, dim=-1), dim=-1)

    def forward(self, student_output, teacher_output, training_match, epoch, step):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        Assumes student_output and teacher_output have shape ViewsXBatchSizeXD
        """
        metrics_dict = {}
        student_temp = self.student_temp

        v_s = training_match["student_views"]
        v_t = training_match["teacher_views"]

        student_out = student_output / student_temp
        student_out = einops.rearrange(student_out, '(v b) f -> v b f',v = v_s)
        student_confidence = F.softmax(student_out, dim=-1).max(-1)[0].mean()

        # teacher centering and sharpening
        teacher_output = einops.rearrange(teacher_output, '(v b) f -> v b f',v = v_t, )
        teacher_out, changed_labels, centering_metrics = self.centering(teacher_output)
        teacher_out, temp = self.apply_teacher_temp(teacher_out, epoch, training_match)
        
        total_loss = 0
        loss_view_count = 0
        acc = 0
        for iq, q in enumerate(teacher_out):
            for ip, p in enumerate(student_out):
                if ip not in training_match["matches"][iq]:
                    continue
                loss = self.loss_func(p, q)
                total_loss = total_loss + loss.mean()
                loss_view_count += 1
                acc = acc + (q.argmax(-1)==p.argmax(-1)).float().mean()
        acc = acc / loss_view_count
        total_loss = total_loss / loss_view_count

        teacher_max, teacher_argmax = teacher_out.max(-1)
        confidence = teacher_max.mean()
        
        all_labels = cat_all_gather(teacher_out.argmax(-1),dim=1)
        student_all_labels = cat_all_gather(student_out.argmax(-1),dim=1)
        unique_labels = all_labels.unique().numel()/self.args.batch_size
        student_unique_labels = student_all_labels.unique().numel()/self.args.batch_size
        target_label_aggreement = (all_labels[0]==all_labels[1]).float().mean()

        self.update_center(teacher_out, step)
        metrics_dict.update(centering_metrics)
        metrics_dict.update({"acc": acc,
                        "changed_labels": changed_labels, 
                        "confidence_tr": confidence, "confidence_st":student_confidence,
                        "teacher_temp": temp, "center_ema": self.center_momentum,
                        "unique_labels_tr": unique_labels, "label_aggreement":target_label_aggreement,
                        "unique_labels_st": student_unique_labels})
        return total_loss, metrics_dict


    @torch.no_grad()
    def update_center(self, output_distribution, step):
        """
        Update center used for teacher output.
        """
        hard_count = F.one_hot(output_distribution.argmax(-1), self.out_dim)
        self.cluster_count+=hard_count.sum([0,1])

        center_update = (hard_count.half()).mean([0,1])
        if distributed.is_initialized():
            distributed.all_reduce(center_update)
            center_update = center_update / distributed.get_world_size()
        self.center = self.center * self.center_momentum + center_update * (1 - self.center_momentum)


    @torch.no_grad()
    def centering(self, teacher_output):
        labels_pre = teacher_output.argmax(-1)
        lower_w, higher_w, = self.centering_modifiers()
        teacher_output = 1-(-teacher_output+1)*lower_w # Decreasing the cosine DISTANCE for small clusters
        teacher_output = (teacher_output+1) * higher_w - 1 # Decreasing the cosine SIMILARITY for large clusters

        teacher_output_argmax = teacher_output.argmax(-1)
        teacher_output_argmax_oh = F.one_hot(teacher_output_argmax, self.out_dim)

        # boosting choices
        bt = torch.ones(teacher_output.shape).type_as(teacher_output)+teacher_output_argmax_oh*0.01
        teacher_output = teacher_output * bt

        teacher_output = torch.clamp(teacher_output, -1, 1)
        labels_post = teacher_output.argmax(-1)
        changed_labels = 1-(labels_post==labels_pre).float().mean()
        return teacher_output, changed_labels, {"llb_min": lower_w.min(), "lhb_min": higher_w.min()}

    def centering_modifiers(self):
        center = F.normalize(self.center,p=1,dim=-1)*self.out_dim
        lower_w = 1 - F.relu(1-center)
        lower_w += torch.rand(lower_w.shape, device=self.center.device)*0.001*(lower_w==0).float() # To prevent from going to 0
        higher_w = 1 - F.relu(1-1/center)
        higher_w += torch.rand(higher_w.shape, device=self.center.device)*0.001*(higher_w==0).float() # To prevent from going to 0
        return lower_w, higher_w

    def apply_teacher_temp(self, teacher_out, epoch, training_match):
        v_t = training_match["teacher_views"]
        teacher_out = teacher_out / self.teacher_temp
        teacher_out = F.softmax(teacher_out,dim=-1).detach()
        return teacher_out, self.teacher_temp

    def centering_metrics(self):
        plt.ioff()
        center, center_ind = self.center.squeeze().sort(dim=-1,descending=False)
        lower_w, higher_w = self.centering_modifiers()
        lower_w = lower_w[0][center_ind]
        higher_w = higher_w[0][center_ind]
        cluster_count = self.cluster_count
        distributed.all_reduce(cluster_count)
        cluster_count, _ = F.normalize(cluster_count,p=1,dim=-1).sort(dim=-1)
        self.cluster_count *= 0
        metrics_dict = {
                "empty_clusters": (cluster_count==0).float().mean(),
                "cluster_size_max": cluster_count.max(),
                "cluster_size_min": cluster_count.min()}
        return metrics_dict

@torch.no_grad()
def purity(clusters, labels):
    cluster_purity = 0
    cluster_count = 0
    for c in clusters.unique():
        c_labels = labels[clusters==c]
        if len(c_labels)>0:
            cluster_count+=1
            unique, counts = torch.unique(c_labels, return_counts=True)
            cluster_purity+=F.normalize(counts.float(),dim=-1,p=1).max()
    return cluster_purity/cluster_count

@torch.no_grad()
def clustering_acc(clusters, labels):
    acc = torch.zeros((1,),device=clusters.device)
    samples_count = torch.zeros((1,),device=clusters.device)
    for c in clusters.unique():
        c_labels = labels[clusters==c]
        if len(c_labels)>0:
            unique, counts = torch.unique(c_labels, return_counts=True)
            label = unique[counts.argmax(-1)]
            acc+= (c_labels==label).float().sum()
            samples_count+=len(c_labels)
    distributed.all_reduce(samples_count)
    distributed.all_reduce(acc)
    acc = acc/samples_count
    return acc

class TrackTrainPseudolabels:
    def __init__(self, clusterings=1):
        self.tracker = [ClusteringPseudolabelTracker() for k in range(clusterings)]

    def add(self, pseudolabels, labels, clustering=0):
        self.tracker[clustering].add(pseudolabels, labels)

    def get_labels_acc(self):
        acc = [tr.get_labels_acc() for tr in self.tracker]
        return acc
    
    def reset(self):
        [tr.reset() for tr in self.tracker]

    def aggregate(self):
        [tr.aggregate() for tr in self.tracker]

    def get_interclustering_nmi(self):
        nmi, count = 0, 0
        for i in range(len(self.tracker)):
            for j in range(i+1, len(self.tracker)):
                nmi+=normalized_mutual_info_score(self.tracker[i].pseudolabels.cpu().numpy(), self.tracker[j].pseudolabels.cpu().numpy())
                count+=1
        nmi = nmi/count
        return nmi

class ClusteringPseudolabelTracker:
    def __init__(self):
        self.pseudolabels = []
        self.labels = []

    def reset(self):
        self.pseudolabels = []
        self.labels = []

    def add(self, pseudolabels, labels):
        pseudolabels = pseudolabels.argmax(-1)
        v = pseudolabels.shape[0]//labels.shape[0]
        b = labels.shape[0]
        labels = labels.unsqueeze(0).repeat(v,1).reshape(-1)
        self.pseudolabels.append(pseudolabels)
        self.labels.append(labels)

    def aggregate(self):
        self.pseudolabels = torch.cat(self.pseudolabels,dim=0)
        self.labels = torch.cat(self.labels,dim=0)
        self.pseudolabels = cat_all_gather(self.pseudolabels)
        self.labels = cat_all_gather(self.labels)

    def get_labels_acc(self):
        acc = clustering_acc(self.pseudolabels, self.labels)
        return acc

