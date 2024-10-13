
import time
import warnings
import torch
import torch.distributed as dist
import wandb
import matplotlib
import matplotlib.pyplot as plt
import os
import os
import sys
import logging
import functools
from termcolor import colored
import argparse
import traceback
from utils.misc import Timer
from utils.misc import load_file

os.environ['WANDB_START_METHOD'] = "thread"
os.environ['WANDB_SILENT'] = "true"
os.environ['WANDB__SERVICE_WAIT'] = "300"
os.environ["WANDB_INIT_TIMEOUT"] = "300"

try:
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
except:
    pass
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True
#plt.switch_backend('agg')

exclude_strings=["torch._dynamo.symbolic_convert", "FakeTensorMode.__torch_dispatch__", "findfont"]

def logger_argparser(args_dict=None):
    parser = argparse.ArgumentParser(conflict_handler='resolve',add_help=False)
    parser.add_argument(
        "--output_dir", default=args_dict.get("output_dir", None), type=str, help="Experiment parent directory")
    parser.add_argument(
        "--entity", default=args_dict.get("entity", None), type=str, help="Wandb entity")
    parser.add_argument(
        "--project_name", default=args_dict.get("project_name", None), type=str, help="Wandb project name")
    parser.add_argument(
        "--run_name", default=args_dict.get("run_name", None), type=str, help="Name of current experiment")
    parser.add_argument(
        "--run_id", default=args_dict.get("run_id", None), type=str, help="ID of current experiment")
    parser.add_argument(
        "--group", default=args_dict.get("group", None), type=str, help="Wandb group")
    parser.add_argument(
        "--tags", default=args_dict.get("tags", None), type=str, help="Wandb tags")
    parser.add_argument(
        "--notes", default=args_dict.get("notes", None), type=str, help="Wandb notes")
    parser.add_argument(
        "--wandb_mode", default=args_dict.get("wandb_mode", "off"), type=str, help="Wandb mode. Possible options: online/offline/off")
    parser.add_argument(
        "--resume", default=args_dict.get("resume"), type=bool, help="Whether to resume run")
    return parser


class Logger:
    def __init__(self, args):

        self.output_dir = args.output_dir
        self.entity = args.entity
        self.project_name = args.project_name or "ML"
        self.run_name = args.run_name or "exp"
        self.group = args.group
        self.tags = args.tags
        self.notes = args.notes
        self.wandb_mode = args.wandb_mode
        self.resume = args.resume
        self.ddp = dist.is_initialized()
        self.timer = Timer()
        self.total_steps = None
        self.start_time = time.time()

        self.config_dict = args.__dict__
        if self.ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 0
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        if self.ddp:
            dist.barrier()

        if hasattr(args, "checkpoint_path") and (args.checkpoint_path is not None and args.__dict__.get("checkpoint_same_run",False)):
            loaded_args = load_file(args.checkpoint_path, args.output_dir)["args"]
            args.run_id = loaded_args.__dict__.get("run_id",None)
            args.run_name = loaded_args.__dict__.get("run_name",None)

        self.args = args
        self.wandb_init()
        if wandb.run is not None:
            self.args.run_path = wandb.run.path
            self.args.run_id = wandb.run._run_id
        

        self.logger = create_logger(self.output_dir, self.rank)
        self.print(f"Initializing experiment {self.run_name}\n")
        self.print(f"Initialized ddp - Global rank: {self.rank} of {self.world_size}")
        self.metric_handler = MetricHandler()
        self.metric_handler_step = MetricHandler()
        self.images_dict = {}

        if self.ddp:
            dist.barrier()
        self.print_args()

    def print_epoch_progress(self, step, of_steps=None, epoch=None, of_epochs=None):
        if step == 0:
            self.epoch_start_time = time.time()
        epoch_time = time.time()-self.epoch_start_time
        steps_per_s = round(step / epoch_time, 2)
        text = self.run_name + " | "
        if epoch is not None:
            text += f"Ep. {epoch}"
            if of_epochs is not None:
                text+=f"/{of_epochs}"
            text+=" - "
        text += f"Step {step}"
        if of_steps is not None:
            text = text + f"/{of_steps}"
        text += f" ({steps_per_s} it/s"
        if of_steps is not None:
            eta_ep = (of_steps-step)/(steps_per_s+0.01)
            text += f", ETA {convert_time(eta_ep)}"
        text += f") |"
        if of_steps is not None:
            prog_ratio = round(step / of_steps * 100, 1)
            text += (int(prog_ratio // 5) * "#" +
                     int(20 - prog_ratio // 5) * " " + "| - ")
        else:
            text += " "
        metrics = self.metric_handler_step.get_avg()
        if len(metrics.keys()) != 0:
            for k, v in metrics.items():
                text += f"{k}={self.rounding(v)} - "
            text = text[:-3]
        self.print(text)
        self.metric_handler_step.reset()

    def print_epoch_end(self, epoch, of_epochs=None):
        """
        Prints epoch-wise progress bar with requested metrics (if they have been logged)
        """
        epoch_time = time.time()-self.epoch_start_time
        text = f"\n{self.run_name} | Ep. {epoch}"
        if of_epochs is not None:
            text += f"/{of_epochs}"
        text += f" | Time: {convert_time(epoch_time)}"
        if of_epochs is not None:
            eta = (time.time()-self.start_time)/(epoch+1)*(of_epochs-epoch-1)
            text += f", ETA {convert_s2dh(eta)}"
        else:
            eta = 0.
        text+=f" | "
        metrics = self.metric_handler.get_avg()
        if len(metrics.keys()) != 0:
            for k, v in metrics.items():
                text += f"{k}={self.rounding(v)} - "
            text = text[:-3]
        text+="\n"
        self.print(text, end='\n')
        return eta

    def epoch_end(self, epoch, of_epochs=None, reset_metrics=True):
        eta = self.print_epoch_end(epoch, of_epochs)
        avg_metrics = self.metric_handler.get_avg()
        avg_metrics["job_eta"] = round(float(eta/3600),2)
        if self.wandb_mode!="off" and self.rank==0:
            wandb.log(avg_metrics, step=epoch)
            wandb.log(self.images_dict, step=epoch)
        if reset_metrics:
            self.metric_handler.reset()
            self.images_dict = {}
        if self.wandb_mode!="off" and self.rank==0:
            self.upload_logs()
        self.metric_handler_step.reset()

    def rounding(self, v):
        """
        Rounds values for printing
        """
        v = float(v)
        if v<0:
            negative=True
            v = -v
        else:
            negative=False
        if v==0:
            return v
        elif v >= 100:
            v = round(v, 1)
        elif v >= 10:
            v = round(v, 2)
        elif v >= 1:
            v = round(v, 4)
        else:
            decimal_count=0
            while v*(10**decimal_count)<1:
                decimal_count+=1
            v = round(v, decimal_count+2)
        if negative:
            v = -v
        return v


    def log(self, metrics:dict, epoch=None):
        metrics_dict = {}
        for k,v in metrics.items():
            if isinstance(v, matplotlib.figure.Figure) or isinstance(v, wandb.Plotly):
                self.images_dict[k]=v
            else:
                if isinstance(v, torch.Tensor):
                    metrics_dict[k]=v.clone().detach()
                else:
                    metrics_dict[k]=v
        self.metric_handler.add_metrics(metrics_dict)
        self.metric_handler_step.add_metrics(metrics_dict)
        
    def log_images(self, img_dict, epoch=None):
        new_dict = {}
        if not (self.wandb_mode=="off" or self.rank!=0 or len(img_dict.keys())==0):
            for k, v in img_dict.items():
                if not isinstance(v, wandb.Plotly):
                    new_dict[k] = wandb.Image(v)
                else:
                    new_dict[k]=v
            wandb.log(new_dict,step=epoch)
        plt.close("all")

    def log_and_write(self, metrics: dict, epoch=None):
        if self.rank==0:
            wandb.log(metrics, step=epoch)

    def print(self, msg, end="\n", only_main_rank=False):
        """
        Handles prints. If lines are to be overwritten, they are printed normally, otherwise they are printed
        via the logger and stored to the corresponding log txt
        """
        if only_main_rank and self.rank != 0:
            return None
        if end == "\r" and self.rank == 0:
            print(self.adjust_print_string_length(msg), end="\r")
        else:
            self.logger.info(msg)

    def info(self, msg):
        self.print(msg)

    def error(self, e):
        msg = traceback.format_exc()
        self.logger.error(msg, exc_info=True)
        self.logger.error(e, exc_info=True)

    def adjust_print_string_length(self, text):
        """
        Adjusts printed messages to fill the terminal and overwrite previous prints on the same line
        """
        try:
            terminal_len = os.get_terminal_size()[0]
        except Exception as e:
            self.print(f"Error in getting actual terminal size: {e}")
            terminal_len = 200
        if terminal_len == 0:
            terminal_len = 200
        if len(text) < terminal_len:
            text = text + (terminal_len - len(text)) * " "
        elif len(text) > terminal_len:
            text = text[:(terminal_len - 5)] + ' ...'
        return text

    def wandb_init(self):
        if self.wandb_mode=="off" or self.rank!=0:
            return None
        wandb_initialized = False
        try:
            wandb.init(project=self.project_name, entity=self.entity, config=self.config_dict, group=self.group,
                    tags=self.tags, notes=self.notes, dir=self.output_dir, id=self.args.run_id if self.args.resume else None,
                    name=self.run_name, resume="allow" if self.args.resume else False, mode=self.wandb_mode, anonymous="allow",
                    settings=wandb.Settings(init_timeout=600))
            wandb_initialized=True
        except:
            self.print("Wandb timeout")

    def print_args(self):
        to_write = ""
        keys = list(self.config_dict.keys())
        keys.sort()
        for k in keys:
            self.print(f"{k} = {self.config_dict[k]}")
            to_write+=f"{k} = {self.config_dict[k]}\n"
        self.print("")
        with open(f"{self.output_dir}/config.txt", "w") as f:
            f.write(to_write)

    def upload_logs(self):
        for rank in range(self.args.world_size):
            try:
                wandb.save(f"{self.output_dir}/log_rank{rank}.txt", base_path=self.output_dir)
            except Exception as e:
                msg = traceback.format_exc()
                self.print(f"Failed to upload log from rank {rank} of {self.args.world_size}")
                self.print(f"Error: {msg}")

    def finish(self, crashed=False):
        if self.wandb_mode!="off" and self.rank==0:
            self.upload_logs()
            if crashed:
                wandb.finish(quiet=True, exit_code=1)
            else:
                wandb.finish()

class MetricHandler:
    def __init__(self):
        self.metrics = {}

    def reset(self):
        self.metrics = {}

    def add_metrics(self, metrics_dict: dict):
        for k, v in metrics_dict.items():
            if k not in self.metrics.keys():
                self.metrics[k] = {"value": v, "count": torch.ones(
                    (1,), device=torch.cuda.current_device())}
            else:
                self.metrics[k]["value"] += v
                self.metrics[k]["count"] += 1

    def get_avg(self):
        metric_averages = {}
        for k, v in self.metrics.items():
            if not isinstance(v["value"], torch.Tensor):
                k_value = torch.tensor(v["value"], device=torch.cuda.current_device())
            else:
                k_value = v["value"].clone()
            k_count = v["count"].clone()
            if dist.is_initialized():
                dist.all_reduce(k_value)
                dist.all_reduce(k_count)
            metric_averages[k] = k_value/k_count
        sorted_dict = {}
        metric_keys = list(metric_averages.keys())
        metric_keys.sort()
        if "loss" in metric_keys:
            sorted_dict["loss"] = metric_averages["loss"]
            metric_keys.remove("loss")
        for k in metric_keys:
            sorted_dict[k] = metric_averages[k]
        return sorted_dict


def update_wandb_run(run_path, metrics_dict):
    api = wandb.Api()
    run = api.run(run_path)
    for k, v in metrics_dict.items():
        run.summary[k] = v
    run.summary.update()
    
@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):

    logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.WARNING)
    logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
    logging.getLogger("torch._inductor").setLevel(logging.WARNING)

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addFilter(NoParsingFilter())
    logger.addFilter(ExcludeFilter(exclude_strings=exclude_strings))
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s] (%(filename)s %(lineno)d): %(message)s'
    color_fmt = colored('[%(asctime)s]', 'green') + \
        colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(
            fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        console_handler.addFilter(ExcludeFilter(exclude_strings=exclude_strings))
        logger.addHandler(console_handler)

    # create file handlers
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(
        output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

class NoParsingFilter(logging.Filter):
    def filter(self, record):
        cut_messages = []
        for cm in cut_messages:
            if cm in record.filename:
                return False
        return True

class ExcludeFilter(logging.Filter):
    def __init__(self, exclude_strings):
        super().__init__()
        self.exclude_strings = exclude_strings

    def filter(self, record):
        for es in self.exclude_strings:
            if es in record.getMessage():
                return False
        return True

def convert_time(t):
    """
    seconds to HH:MM string format
    """
    h = int(t // 3600)
    m = int((t - h * 3600) // 60)
    s = str(h).zfill(2) + ":" + str(m).zfill(2)
    return s

def convert_s2dh(t):
    h = t/3600
    if h<24:
        return f"{round(h)}h"
    else:
        d = h//24
        h = h%24
        return f"{round(d)}d{round(h)}h"