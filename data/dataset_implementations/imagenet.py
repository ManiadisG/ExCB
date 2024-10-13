from data.dataset_classes import ImageDataset
import os
import glob
import numpy as np
from scipy import io
from data.transforms import *
from utils.misc import export_fn
import urllib

norm_mean = (0.485, 0.456, 0.406)
norm_std = (0.228, 0.224, 0.225)

@export_fn
def build_imagenet(dataset_path, data_set, args, transforms=None):
    if transforms is None:
        if data_set=="train":
            transforms = get_train_transforms(args.crop_min_scale, args.crop_size,
                                            args.p_flip, args.p_colorjitter, args.s_colorjitter,
                                            args.p_gray, args.p_blur, args.p_solarize, args.mini_crops,
                                            args.mini_crop_min_scale, args.mini_crop_size, norm_mean, norm_std, 
                                            args.teacher_transform_views)
        elif data_set=="val" or data_set=="train_inference":
            transforms = get_val_transforms(args.crop_size, args.crop_resize, norm_mean, norm_std)
        elif data_set=="linear_train":
            transforms = get_linear_train_transforms(args.crop_size, norm_mean, norm_std)
        elif data_set=="linear_eval":
            transforms = get_linear_eval_transforms(args.crop_size, args.crop_resize, norm_mean, norm_std)
        elif data_set in ["knn_eval", "knn_train"]:
            transforms = get_knn_transforms(args.crop_size, args.crop_resize, norm_mean, norm_std)
        elif data_set=="semi_sup_train":
            transforms = get_ssup_train_transforms(args.crop_size, norm_mean, norm_std)
        elif data_set=="semi_sup_eval":
            transforms = get_ssup_eval_transforms(args.crop_size, args.crop_resize, norm_mean, norm_std)
    return ImageNetDataset(dataset_path, data_set, transforms=transforms, version=args.dataset_version)

class ImageNetDataset(ImageDataset):
    def __init__(self, dataset_path="./datasets/ImageNet", data_set="train", version="default", transforms=None):
        train_data, train_annotations, val_data, val_annotations = get_imagenet(dataset_path, version)
        if "train" in data_set:
            data, annotations = train_data, train_annotations
        else:
            data, annotations = val_data, val_annotations
        if transforms is None:
            transforms = get_normalize(norm_mean, norm_std)
        super(ImageNetDataset, self).__init__(data, annotations, transforms)


def get_imagenet(dataset_path=None, version="default"):
    if dataset_path is None:
        dataset_path = "./datasets/ImageNet"
    return get_imagenet_file_reading(dataset_path, version)

def _parse_val_groundtruth_txt(devkit_root):
    file = os.path.join(devkit_root, "data",
                        "ILSVRC2012_validation_ground_truth.txt")
    with open(file, 'r') as txtfh:
        val_idcs = txtfh.readlines()
    return [int(val_idx) - 1 for val_idx in val_idcs]


def _parse_meta_mat(devkit_root):
    metafile = os.path.join(devkit_root, "data", "meta.mat")
    meta = io.loadmat(metafile, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
            if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(', ')) for clss in classes]
    idx_to_wnid = {idx - 1: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_idx = {wnid: idx - 1 for idx, wnid in zip(idcs, wnids)}
    wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}

    return idx_to_wnid, wnid_to_idx, wnid_to_classes


def get_imagenet_file_reading(dataset_path, version):
    if os.path.exists(dataset_path + '/ILSVRC2012_devkit_t12'):
        idx_to_wnid_default, wnid_to_idx_default, wnid_to_classes_default = _parse_meta_mat(dataset_path + '/ILSVRC2012_devkit_t12')
    else:
        idx_to_wnid_default, wnid_to_idx_default, wnid_to_classes_default = {}, {}, {}
        wnids = os.listdir(dataset_path + '/train/')
        wnids.sort()
        for i, wnid in enumerate(wnids):
            idx_to_wnid_default[i]=wnid
            wnid_to_idx_default[wnid]=i
            wnid_to_classes_default[wnid]=str(f"class_{i}")
            
    if version.lower() in ["10perc", "1perc"]:
        perc = 1 if version.lower()=="1perc" else 10
        subset_file = urllib.request.urlopen(f"https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{perc}percent.txt")
        list_imgs = [li.decode("utf-8").split('\n')[0] for li in subset_file]
    else:
        list_imgs = None
        
    if version.lower() not in ["default", "imagenet", "1perc", "10perc"]:
        if version == "imagenet-dogs":
            c_folder = ["n02085936", "n02086646", "n02088238", "n02091467", "n02097130", "n02099601", "n02101388",
                        "n02101556", "n02102177", "n02105056", "n02105412", "n02105855", "n02107142", "n02110958",
                        "n02112137"]
        elif version=="imagenet-10":
            c_folder = ["n02056570", "n02085936", "n02128757", "n02690373", "n02692877", "n03095699", "n04254680",
                        "n04285008", "n04467665", "n07747607"]
        elif version=="imagenet-100":
            c_folder = ["n02869837", "n01749939", "n02488291", "n02107142", "n13037406", "n02091831", "n04517823", "n04589890", 
                        "n03062245", "n01773797", "n01735189", "n07831146", "n07753275", "n03085013", "n04485082", "n02105505", 
                        "n01983481", "n02788148", "n03530642", "n04435653", "n02086910", "n02859443", "n13040303", "n03594734", 
                        "n02085620", "n02099849", "n01558993", "n04493381", "n02109047", "n04111531", "n02877765", "n04429376", 
                        "n02009229", "n01978455", "n02106550", "n01820546", "n01692333", "n07714571", "n02974003", "n02114855", 
                        "n03785016", "n03764736", "n03775546", "n02087046", "n07836838", "n04099969", "n04592741", "n03891251", 
                        "n02701002", "n03379051", "n02259212", "n07715103", "n03947888", "n04026417", "n02326432", "n03637318", 
                        "n01980166", "n02113799", "n02086240", "n03903868", "n02483362", "n04127249", "n02089973", "n03017168", 
                        "n02093428", "n02804414", "n02396427", "n04418357", "n02172182", "n01729322", "n02113978", "n03787032", 
                        "n02089867", "n02119022", "n03777754", "n04238763", "n02231487", "n03032252", "n02138441", "n02104029", 
                        "n03837869", "n03494278", "n04136333", "n03794056", "n03492542", "n02018207", "n04067472", "n03930630", 
                        "n03584829", "n02123045", "n04229816", "n02100583", "n03642806", "n04336792", "n03259280", "n02116738", 
                        "n02108089", "n03424325", "n01855672", "n02090622"]
        idx_to_wnid_, wnid_to_idx_, wnid_to_classes_ = {},{},{}
        for i,c_ in enumerate(c_folder):
            idx_to_wnid_[i]=c_
            wnid_to_idx_[c_]=i
            wnid_to_classes_[c_]=wnid_to_classes_default[c_]
        idx_to_wnid, wnid_to_idx, wnid_to_classes = idx_to_wnid_, wnid_to_idx_, wnid_to_classes_
    else:
        idx_to_wnid, wnid_to_idx, wnid_to_classes = idx_to_wnid_default, wnid_to_idx_default, wnid_to_classes_default
        c_folder = list(wnid_to_idx.keys())
        

    if list_imgs is not None:
        list_imgs_per_wnid = {}
        for wnid in wnid_to_idx.keys():
            list_imgs_per_wnid[wnid] = [img for img in list_imgs if wnid in img]

    train_path = dataset_path + '/train/'
    train_samples, train_labels = [], []
    for k in wnid_to_classes.keys():
        k_samples = glob.glob(train_path + k + '/*')
        if list_imgs is not None:
            k_samples = [sample for sample in k_samples if sample.split("/")[-1] in list_imgs_per_wnid[k]]
        train_samples += k_samples
        train_labels += len(k_samples) * [wnid_to_idx[k]]

    val_path = dataset_path + '/val'
    if os.path.exists(dataset_path + '/ILSVRC2012_devkit_t12'):
        val_labels = _parse_val_groundtruth_txt(dataset_path + '/ILSVRC2012_devkit_t12')
        val_samples = glob.glob(val_path + '/*')
        val_samples.sort()
        if version!="default":
            val_labels_, val_samples_ = [], []
            for sample, label in zip(val_samples, val_labels):
                wnid = idx_to_wnid_default[label]
                if wnid not in c_folder:
                    continue
                val_samples_.append(sample)
                val_labels_.append(wnid_to_idx[wnid])
            val_labels, val_samples = val_labels_, val_samples_
    else:
        if version=="default":
            val_samples = glob.glob(val_path + '/*/*')
            val_samples.sort()
            val_labels = []
            for i, vs in enumerate(val_samples):
                wnid = vs.split("/")[-2]
                val_labels.append(wnid_to_idx[wnid])
        else:
            val_samples_ = glob.glob(val_path + '/*/*')
            val_samples_.sort()
            val_labels, val_samples = [], []
            for i, vs in enumerate(val_samples_):
                wnid = vs.split("/")[-2]
                if wnid not in c_folder:
                    continue
                val_samples.append(vs)
                val_labels.append(wnid_to_idx[wnid])

    return np.array(train_samples), np.array(train_labels), np.array(val_samples), np.array(val_labels)
