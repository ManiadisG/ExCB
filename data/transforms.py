
import torch
from torchvision import transforms as T
from timm.data.auto_augment import rand_augment_transform
from timm.data.random_erasing import RandomErasing
from utils.misc import export_fn


@export_fn
def get_normalize(norm_mean = (0.485, 0.456, 0.406), norm_std = (0.228, 0.224, 0.225)):
    return T.Compose([T.ToTensor(), T.Normalize(mean=norm_mean, std=norm_std)])

@export_fn
def get_val_transforms(crop_size=224, crop_resize=256, norm_mean = (0.485, 0.456, 0.406), norm_std = (0.228, 0.224, 0.225)):
    return T.Compose([T.Resize(crop_resize), T.CenterCrop(crop_size),
                     T.ToTensor(), T.Normalize(mean=norm_mean, std=norm_std)])

@export_fn
def get_train_transforms(crop_min_scale=0.2, crop_size=224,
                         p_flip=0.5, p_colorjitter=0.8, s_colorjitter = 0.5,
                         p_gray = 0.2, p_blur = [1., 0.1], p_solarize = [0., 0.2],
                         mini_crops=0, mini_crop_min_scale=0.05, mini_crop_size=96, 
                         norm_mean = (0.485, 0.456, 0.406), norm_std = (0.228, 0.224, 0.225), 
                         teacher_transform_views=2):
    assert len(p_blur)==len(p_solarize)
    return MultiCropTransforms(crop_min_scale, crop_size,
                         p_flip, p_colorjitter, s_colorjitter,
                         p_gray, p_blur, p_solarize, mini_crops, mini_crop_min_scale, mini_crop_size,
                         norm_mean, norm_std, teacher_transform_views)

@export_fn
def get_linear_train_transforms(crop_size=224, norm_mean = (0.485, 0.456, 0.406), norm_std = (0.228, 0.224, 0.225)):
    return T.Compose([T.RandomResizedCrop(crop_size), T.RandomHorizontalFlip(), get_normalize(norm_mean, norm_std)])

@export_fn
def get_linear_eval_transforms(crop_size=224, crop_resize=256,
                                norm_mean = (0.485, 0.456, 0.406), norm_std = (0.228, 0.224, 0.225)):
    return T.Compose([T.Resize(crop_resize), T.CenterCrop(crop_size), get_normalize(norm_mean, norm_std)])

@export_fn
def get_knn_transforms(crop_size=224, crop_resize=256,
                                norm_mean = (0.485, 0.456, 0.406), norm_std = (0.228, 0.224, 0.225)):
    return T.Compose([T.Resize(crop_resize), T.CenterCrop(crop_size), get_normalize(norm_mean, norm_std)])

@export_fn
def get_ssup_train_transforms(crop_size=224, norm_mean = (0.485, 0.456, 0.406), norm_std = (0.228, 0.224, 0.225)):
    return T.Compose([T.RandomResizedCrop(crop_size), T.RandomHorizontalFlip(), get_normalize(norm_mean, norm_std)])

@export_fn
def get_ssup_eval_transforms(crop_size=224, crop_resize=256,
                                norm_mean = (0.485, 0.456, 0.406), norm_std = (0.228, 0.224, 0.225)):
    return T.Compose([T.Resize(crop_resize), T.CenterCrop(crop_size), get_normalize(norm_mean, norm_std)])

class MultiCropTransforms(object):
    def __init__(self, crop_min_scale=0.2, crop_size=224,
                 p_flip=0.5, p_colorjitter=0.8, s_colorjitter = 0.5,
                 p_gray = 0.2, p_blur = [1., 0.1], p_solarize = [0., 0.2],
                 mini_crops=0, mini_crop_min_scale=0.05, mini_crop_size=96,
                 norm_mean = (0.485, 0.456, 0.406), norm_std = (0.228, 0.224, 0.225), 
                 teacher_transform_views=2):
        assert len(p_blur)==len(p_solarize)
        self.crop_min_scale=crop_min_scale
        self.crop_size=crop_size
        self.p_flip=p_flip
        self.p_colorjitter=p_colorjitter
        self.s_colorjitter=[0.8*s_colorjitter, 0.8*s_colorjitter, 0.4*s_colorjitter, 0.2*s_colorjitter]
        self.p_gray=p_gray
        self.p_blur=p_blur
        self.p_solarize=p_solarize
        self.norm_mean=norm_mean
        self.norm_std=norm_std
        self.mini_crops=mini_crops
        self.mini_crop_min_scale=mini_crop_min_scale
        self.mini_crop_size=mini_crop_size
        self.teacher_transform_views = teacher_transform_views

        self.set_up_transforms(crop_min_scale)
    

    def set_up_transforms(self, crop_min_scale):
        self.transforms = []
        self.mini_transforms = None
        self.teacher_transforms = None
        for pblur, psolarize in zip(self.p_blur, self.p_solarize):
            rrc = T.RandomResizedCrop(self.crop_size, scale=(crop_min_scale, 1), interpolation=T.InterpolationMode.BICUBIC)
            rhf = T.RandomHorizontalFlip(p=self.p_flip)
            cj = T.RandomApply([T.ColorJitter(brightness=self.s_colorjitter[0], contrast=self.s_colorjitter[1],
                                              saturation=self.s_colorjitter[2], hue=self.s_colorjitter[3])], p=0.8)
            rg = T.RandomGrayscale(p=0.2)
            gb = GaussianBlur(p=pblur)
            so = T.RandomSolarize(threshold=128, p=psolarize)
            no = get_normalize(self.norm_mean, self.norm_std)
            self.transforms.append(T.Compose([rrc, rhf, cj, rg, gb, so, no]))

        if self.mini_crops!=0:
            rrc = T.RandomResizedCrop(self.mini_crop_size, scale=(self.mini_crop_min_scale, crop_min_scale), interpolation=T.InterpolationMode.BICUBIC)
            rhf = T.RandomHorizontalFlip(p=self.p_flip)
            cj = T.RandomApply([T.ColorJitter(brightness=self.s_colorjitter[0], contrast=self.s_colorjitter[1],
                                              saturation=self.s_colorjitter[2], hue=self.s_colorjitter[3])], p=0.8)
            rg = T.RandomGrayscale(p=0.2)
            gb = GaussianBlur(p=self.p_blur[0])
            no = get_normalize(self.norm_mean, self.norm_std)
            self.mini_transforms = T.Compose([rrc, rhf, cj, rg, gb, no])

    def __call__(self, img):
        to_return = torch.stack([tr(img) for tr in self.transforms])
        if self.teacher_transforms is not None or self.mini_transforms is not None:
            to_return = [to_return]
        if self.teacher_transforms is not None:
            to_return.append(torch.stack([self.teacher_transforms(img) for tr in range(self.teacher_transform_views)]))
        if self.mini_transforms is not None:
            to_return.append(torch.stack([self.mini_transforms(img) for i in range(self.mini_crops)]))
        return to_return

class GaussianBlur(T.RandomApply):
    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = T.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)