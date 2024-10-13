import sys
import math
import time
import os
import torch
import torch.distributed as dist
import numpy as np
from scipy.sparse import csr_matrix
import torch.nn as nn
import pickle
import copy
from utils.ddp_utils import cat_all_gather


def split_to_intervals(end_value, intervals, start_value=0):
    chunk_size = (end_value - start_value) / intervals
    intervals = [math.floor(start_value + ch * chunk_size) for ch in range(intervals)]
    intervals.append(end_value)
    return intervals

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def export_fn(fn):
    """
    Implementation adapted from https://stackoverflow.com/a/41895257
    """
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        name = fn.__name__
        all_ = mod.__all__
        assert name not in mod.__all__
        all_.append(name)
    else:
        mod.__all__ = [fn.__name__]
    return fn

class Timer:
    def __init__(self):
        self.times_tic = {}
        self.times_toc = {}

    def get_divider(self, unit):
        if unit=="minutes":
            return 60
        elif unit=="hours":
            return 3600
        else:
            return 1
            
    def tic(self, names=None):
        if names is None:
            names=["time"]
        elif isinstance(names, str):
            names = [names]
        for name in names:
            self.times_tic[name]=time.time()

    def toc(self, names=None, reset=False):
        if names is None:
            names=["time"]
        elif isinstance(names, str):
            names = [names]
        for name in names:
            if name in self.times_tic.keys() and self.times_tic[name] is not None:
                toc=time.time()-self.times_tic[name]
                if reset or name not in self.times_toc.keys():
                    self.times_toc[name]=toc
                else:
                    self.times_toc[name]+=toc
       
    def reset(self):
        self.times_tic = {}
        self.times_toc = {}

    def get_time(self, name=None, relative_time=1., reset=False):
        if isinstance(relative_time, str):
            if relative_time in list(self.times_toc.keys()):
                relative_time = self.times_toc[relative_time]
            else:
                relative_time = 1.
        if name is None:
            times = copy.deepcopy(self.times_toc)
            for n in times.keys():
                times[n]=times[n]/relative_time
            to_return=times
        else:
            assert name in self.times_toc.keys()
            to_return=self.times_toc[name]
        if reset:
            self.reset()
        return to_return


def scheduling(max_value, min_value, step, of_steps, schedule="cosine", decreasing=True):
    assert schedule in ["cosine", "linear"]
    if decreasing:
        if schedule=="cosine":
            return min_value + 0.5 * (max_value - min_value) * (1 + math.cos(math.pi * step / of_steps))
        elif schedule=="linear":
            return max_value-(max_value-min_value) * step/of_steps
    else:
        if schedule=="cosine":
            return max_value + 0.5 * (min_value - max_value) * (1 + math.cos(math.pi * step / of_steps))
        elif schedule=="linear":
            return (max_value-min_value) * step/of_steps + min_value

def load_file(file_path, home_dir=None, logger=None):
    def lcprint(msg):
        if logger is None:
            print(msg)
        else:
            logger.print(msg)
    found_path=None
    if os.path.exists(file_path):
        found_path = file_path
    else:
        lcprint(f"File not found at {file_path}")
        if home_dir is not None:
            file_path = f"{home_dir}/{file_path}"
            if os.path.exists(file_path):
                found_path = file_path
            else:
                lcprint(f"File not not found at {file_path}")
    map_location = "cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count())
    return torch.load(found_path, map_location=map_location)

def get_local_assignments(features, centroids):
    def _get_assignments(features, centroids):
        dot_products = torch.mm(features, centroids.t())
        _, local_assignments = dot_products.max(dim=1)
        return local_assignments
    fc, fc_max = 0, len(features)
    assignments = []
    while fc<fc_max:
        assignments.append(_get_assignments(features[fc:(fc+25000)], centroids))
        fc+=25000
    return torch.cat(assignments,dim=0)


@torch.no_grad()
def distributed_kmeans(teacher_data, student_data, num_clusters, args, logger, nmb_kmeans_iters=50, complementary_data=None):
    feat_dim = teacher_data.shape[-1]
    while True:
        # init centroids with elements from memory bank of rank 0
        centroids = torch.empty(num_clusters,feat_dim).cuda(non_blocking=True).type_as(teacher_data)
        student_centroids = torch.empty(num_clusters,feat_dim).cuda(non_blocking=True).type_as(teacher_data)
        assert len(teacher_data)>(num_clusters//dist.get_world_size())
        random_idx = torch.randperm(len(teacher_data))[:(num_clusters//dist.get_world_size())]
        centroids = teacher_data[random_idx]
        centroids = cat_all_gather(centroids)
        dist.broadcast(centroids, 0)

        for n_iter in range(nmb_kmeans_iters + 1):
            # E step
            local_assignments = get_local_assignments(teacher_data, centroids)
            #dot_products = torch.mm(teacher_data, centroids.t())
            #_, local_assignments = dot_products.max(dim=1)
            # finish
            if n_iter == nmb_kmeans_iters:
                break
            # M step
            centroids = get_centroids(teacher_data, centroids, local_assignments, num_clusters, feat_dim)
            # normalize centroids
            centroids = nn.functional.normalize(centroids, dim=1, p=2)
        global_assignments = cat_all_gather(local_assignments)
        if len(global_assignments.unique())==num_clusters:
            break
        else:
            logger.print("Unstable clustering, repeating clustering")
    student_centroids = get_centroids(student_data, student_centroids, local_assignments, num_clusters, feat_dim)
    return centroids, student_centroids, global_assignments

def get_centroids(features, centroids, local_assignments, num_clusters, feat_dim):
    where_helper = get_indices_sparse(local_assignments.cpu().numpy())
    counts = torch.zeros(num_clusters).cuda(non_blocking=True).int()
    emb_sums = torch.zeros(num_clusters, feat_dim).cuda(non_blocking=True).type_as(features)
    for k in range(len(where_helper)):
        if len(where_helper[k][0]) > 0:
            emb_sums[k] = torch.sum(features[where_helper[k][0]], dim=0,)
            counts[k] = len(where_helper[k][0])
    dist.all_reduce(counts)
    mask = counts > 0
    dist.all_reduce(emb_sums)
    centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)
    return centroids

def get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]
      
def resnet_state_dict_check(model_checkpoint):
    bad_names = ['layer0.0.weight', 'layer0.1.weight', 'layer0.1.bias', 'layer0.1.running_mean', 'layer0.1.running_var', 'layer0.1.num_batches_tracked']
    good_names = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked']
    for bn, gn in zip(bad_names, good_names):
        if bn in model_checkpoint.keys():
            model_checkpoint[gn] = model_checkpoint[bn]
            del model_checkpoint[bn]
    return model_checkpoint