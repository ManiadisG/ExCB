from utils.logger import Logger
from utils.misc import export_fn, scheduling, load_file, distributed_kmeans
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from engine.model import EMAHandler
import einops
from utils.misc import Timer, scheduling
from engine.optimizer import clip_gradients


@export_fn
class Trainer:
    def __init__(self, student, teacher, criterion, optimizer, scheduler, args, logger:Logger):
        self.train_step=0
        self.epoch=0
        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.logger = logger

        self.EMAHandler = EMAHandler(student, teacher, args.checkpoint_path is not None)
        self.scaler = GradScaler()
        self.mixed_precision = args.__dict__.get("mixed_precision", True)
        self.logger.print(f"Mixed precision: {'ON' if self.mixed_precision else 'OFF'}")

        self.timer = Timer()
        self.ks = 0
        self.training_match = None

    def save_checkpoint(self, path=None):
        if self.args.rank==0:
            state_dict = {"student": self.student.module.state_dict(),
                          "teacher":self.teacher.module.state_dict(),
                          "criterion": self.criterion.state_dict(),
                          "optimizer":self.optimizer.state_dict(),
                          "scaler":self.scaler.state_dict(),
                          "scheduler": self.scheduler.state_dict(),
                          "args":self.args,
                          "ks": self.ks,
                          "train_step":self.train_step,
                          "epoch":self.epoch,
                          }
            if path is None:
                path = f"{self.args.output_dir}/checkpoint.pth"
            torch.save(state_dict, path)



    def load_checkpoint(self, path, strict=True):
        checkpoint = load_file(path, self.args.output_dir, self.logger)
        self.load_state_dict(self.criterion, checkpoint["criterion"], "criterion", strict)
        self.load_state_dict(self.optimizer, checkpoint["optimizer"], "optimizer", strict)
        self.load_state_dict(self.scheduler, checkpoint["scheduler"], "scheduler", strict)
        self.load_state_dict(self.scaler, checkpoint["scaler"], "scaler", strict)
        self.train_step = checkpoint["train_step"]
        self.epoch = checkpoint["epoch"]
        self.ks = checkpoint["ks"]
        self.logger.print(f"Checkpoint loaded from {path}")
        
    def train_epoch(self, dataloader, print_interval=None):
        
        self.timer.tic("t_epoch")
        if print_interval is None:
            print_interval = max(len(dataloader)//self.args.prints_per_epoch, 5)
        epoch_steps = len(dataloader)
        
        if self.epoch==0:
            self.init_centroids(self.args.num_clusters, dataloader, clustering_samples=self.args.clustering_samples)

        dataloader.sampler.set_epoch(self.epoch)

        if self.args.crop_min_scale_final>0:
            min_crop_scale = scheduling(self.args.crop_min_scale, self.args.crop_min_scale_final, self.epoch, self.args.epochs)
            dataloader.dataset.transformations.set_up_transforms(min_crop_scale)
        else:
            min_crop_scale = self.args.crop_min_scale
            

        self.timer.tic("t_dataloading")
        for batch_id, batch in enumerate(dataloader):
            self.optimizer.zero_grad()

            self.timer.toc("t_dataloading")
            self.timer.tic("t_step_prep")

            lr, wd = self.scheduler.step()
            wd, wd_centroid = self.optimizer.param_groups[0]["weight_decay"], \
                                      self.optimizer.param_groups[-1]["weight_decay"]
            ema_momentum = scheduling(self.args.ema_end_value, self.args.ema_momentum, self.train_step, epoch_steps*self.args.epochs, decreasing=False)

            idx, samples, annotations = batch
            teacher_samples, student_samples, training_match = self.unroll_batch(samples)
            self.timer.toc("t_step_prep")
            self.timer.tic("t_forward")
            metrics = {}

            if self.mixed_precision:
                with autocast(self.mixed_precision):
                    loss, metrics_list, teacher_similarities = self.train_forward(teacher_samples, student_samples, training_match)
                self.scaler.scale(loss).backward()
                if self.args.clip_grad>0:
                    self.scaler.unscale_(self.optimizer)
                    clip_gradients(self.student, self.args.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, metrics_list, teacher_similarities = self.train_forward(teacher_samples, student_samples, training_match)
                loss.backward()
                if self.args.clip_grad>0:
                    clip_gradients(self.student, self.args.clip_grad)
                self.optimizer.step()
            self.EMAHandler.ema_step(self.student, self.teacher, ema_momentum)

            metrics.update(metrics_list)
            self.timer.toc("t_forward")
            self.timer.tic("t_step_end")

            if loss.isnan():
                self.logger.print("NaN loss, saving state and raising error")
                self.save_checkpoint(f"{self.args.output_dir}/checkpoint_NaN.pth")
                raise ValueError("NaN loss")

            metrics.update({"loss":loss, "ema_momentum": ema_momentum, "lr": lr, "wd": wd, "wd_centroid": wd_centroid})
            self.logger.log(metrics, epoch=self.epoch)

            if print_interval>0 and batch_id%print_interval==0:
                self.logger.print_epoch_progress(batch_id, epoch_steps, self.epoch, self.args.epochs)
            self.train_step+=1
            self.timer.toc("t_step_end")
            self.timer.tic("t_dataloading")

        self.logger.log({"crop_min_scale": min_crop_scale})
        self.logger.log(self.criterion.centering_metrics(),epoch=self.epoch)
        self.optimizer.zero_grad()
        
        self.timer.toc("t_epoch")
        self.logger.log(self.timer.get_time(relative_time="t_epoch", reset=True))
        
    def train_forward(self, teacher_samples, student_samples, training_match):
        with torch.no_grad():
            teacher_similarities = self.teacher([teacher_samples])
        student_similarities = self.student(student_samples, get_predictor=True)
        loss, metrics_list = self.criterion(student_similarities, teacher_similarities, training_match, self.epoch, self.train_step)
        return loss, metrics_list, teacher_similarities

    def unroll_batch(self, samples):
        mini_crops = self.args.mini_crops!=0
        
        if mini_crops: # samples = [student_view, mini_crops]
            student_samples = [samples[0], samples[1]]
            student_views = [samples[0].shape[1], samples[1].shape[1]]
            teacher_samples = samples[0]
        else : # samples = student_view
            student_samples = [samples]
            student_views = [samples.shape[1]]
            teacher_samples = samples

        for i in range(len(student_samples)):
            student_samples[i] = einops.rearrange(student_samples[i], "b v c h w -> (v b) c h w").to(self.args.gpu,non_blocking=True)

        teacher_views = teacher_samples.shape[1]
        teacher_samples = einops.rearrange(teacher_samples, "b v c h w -> (v b) c h w").to(self.args.gpu,non_blocking=True)

        if self.training_match is None:
            f = 2 if self.args.predictor else 1
            matches = []
            for tv in range(teacher_views):
                matches.append([tv_ for tv_ in range(teacher_views) if tv!=tv_])
                matches[tv]+=[tv_+sum(student_views) for tv_ in range(sum(student_views))]
            self.training_match = {"student_views": sum(student_views)*f, "teacher_views":teacher_views, "matches": matches}

        return teacher_samples, student_samples, self.training_match

    torch.no_grad()
    def init_centroids(self, num_clusters, dataloader, student_weights=False, clustering_samples=60000):
        torch.cuda.empty_cache()
        dataloader.sampler.set_epoch(self.args.epochs)
        world_size = dist.get_world_size()
        features_limit = clustering_samples
        features_count, print_count = 0, 0
        teacher_features, student_features = [], []
        self.logger.print(f"Initializing centroids.")
        for batch_id, batch in enumerate(dataloader):
            if features_count>(features_limit/20*print_count):
                self.logger.print(f"Feature extraction: {features_count} of {features_limit}")
                print_count+=1
            idx, samples, annotations = batch
            teacher_samples, student_samples, training_match = self.unroll_batch(samples)
            c_view = teacher_samples[(len(teacher_samples)//2)*(batch_id%2):(len(teacher_samples)//2)*(batch_id%2+1)]
            with torch.no_grad():
                with autocast(self.mixed_precision):
                    teacher_output = self.teacher.module.get_projections(c_view).detach()
                    if student_weights:
                        student_output = self.student.module.get_projections(c_view).detach()
                    else:
                        student_output = teacher_output
                teacher_features.append(teacher_output)
                student_features.append(student_output)
            features_count+=len(teacher_output) * world_size
            if features_count>=features_limit:
                break
        teacher_features = torch.cat(teacher_features)
        student_features = torch.cat(student_features)
        self.logger.print(f"Running k-means")
        teacher_centroids, student_centroids, global_assignments = distributed_kmeans(teacher_features, student_features, num_clusters, self.args, self.logger)
        self.logger.print(f"Finished k-means")
        self.teacher.module.classifier.centroid_layer.weight.data = teacher_centroids.data
        self.teacher.module.classifier.centroid_layer.weight_v.data = teacher_centroids.data
        self.student.module.classifier.centroid_layer.weight.data = student_centroids.data
        self.student.module.classifier.centroid_layer.weight_v.data = student_centroids.data
        torch.cuda.empty_cache()

    def load_state_dict(self, nn_module, state_dict, name, strict=False):
        if isinstance(nn_module, torch.nn.Module):
            r = nn_module.load_state_dict(state_dict, strict=strict)
        else:
            r = nn_module.load_state_dict(state_dict)
        self.logger.print(f"Loaded state for module {name}")
        self.logger.print(f"Inconsistencies: {r}")

