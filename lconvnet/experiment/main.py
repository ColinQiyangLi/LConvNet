import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from lconvnet.utils import get_hms, Accumulator, Logger, Streamline
from lconvnet.layers.core import l2_lipschitz_constant_checker

from tensorboardX import SummaryWriter
from shutil import copyfile, rmtree
import os, sys, time, yaml, tempfile, uuid

from datetime import datetime


def larger_is_better(best_v, new_v, key=None):
    if best_v is None:
        return True
    if key is not None:
        return True if new_v(key) > best_v(key) else False
    return True if new_v > best_v else False


def default_accuracy_comparator(best_v, new_v):
    return larger_is_better(best_v, new_v, "accuracy")


def default_loss_comparator(best_v, new_v):
    if best_v is None:
        return True
    return not larger_is_better(best_v, new_v, "loss")


class Experiment:
    def __init__(
        self,
        trainer,
        dataloaders,
        num_epochs,
        train_batch_size,
        test_batch_size,
        exp_dir=None,
        best_metrics_comparator=default_accuracy_comparator,
        post_steps={},
        seed=0,
    ):

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.trainloader, self.testloader, self.mini_testloader = dataloaders(
            train_batch_size=train_batch_size, test_batch_size=test_batch_size
        )
        self.trainer = trainer
        self.exp_dir = exp_dir

        self.epoch = 1
        self.best_metrics_comparator = best_metrics_comparator
        self.best_metrics = None
        self.num_epochs = num_epochs
        self.post_steps = post_steps

    def register_exp_dir(self, exp_dir):
        assert self.exp_dir is None
        self.exp_dir = exp_dir

    def save_state(self):
        state = {
            "trainer": self.trainer.state_dict(),
            "best_metrics": self.best_metrics,
            "epoch": self.epoch,
        }
        save_point = os.path.join(
            self.ckpt_dir, "{}.pth".format(self.trainer.net.__class__.__name__)
        )
        torch.save(state, save_point)

    def load_state(self):
        save_point = os.path.join(
            self.ckpt_dir, "{}.pth".format(self.trainer.net.__class__.__name__)
        )
        state = torch.load(save_point)
        self.trainer.load_state_dict(state["trainer"])
        if "best_metrics" in state:  # backward compatibility
            self.best_metrics = state["best_metrics"]
        self.epoch = state["epoch"]

    def log_scalars(self, scalar_dict, is_training):
        for name, value in scalar_dict.items():
            prefix = "train" if is_training else "valid"
            self.writer.add_scalars(name, {prefix: value}, self.epoch)

    def log_histogram(self, histogram_dict):
        for name, value in histogram_dict.items():
            self.writer.add_histogram(name, value.numpy(), self.epoch)

    def log_acc(self, acc_train, acc_valid):
        for is_training, acc in zip([True, False], [acc_train, acc_valid]):
            scalar_dict = acc.filter(dtype="scalar")
            self.log_scalars(scalar_dict, is_training)
        histogram_dict = acc.filter(dtype="histogram")
        self.log_histogram(histogram_dict)

    def launch(self, resume=False, cfg_path_source=None, test_only=False, tag="norm"):
        time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        self.log_dir = os.path.join(self.exp_dir, "logs")
        self.cfg_path = os.path.join(self.exp_dir, "cfg.yaml")
        self.res_dir = os.path.join(
            self.exp_dir, "results-{tag}-{time}.yaml".format(tag=tag, time=time_str)
        )
        exists = os.path.exists(self.exp_dir)
        if not exists or not resume:
            if exists:
                print("Removing previous experiment...")
                id1 = uuid.uuid1()

                temp_dir = tempfile.gettempdir()
                copyfile(cfg_path_source, os.path.join(temp_dir, str(id1) + "cfg.yaml"))
                rmtree(self.exp_dir)
                os.makedirs(self.ckpt_dir)
                copyfile(os.path.join(temp_dir, str(id1) + "cfg.yaml"), cfg_path_source)
            else:
                os.makedirs(self.ckpt_dir)

        self.log_text_dir = os.path.join(
            self.exp_dir, "output-{tag}-{time}.log".format(tag=tag, time=time_str)
        )
        sys.stdout = Logger(self.log_text_dir)
        if hasattr(self, "__config__"):
            print("Launching experiment with the configuration description:")
            print(yaml.dump(self.__config__))
        self.writer = SummaryWriter(self.log_dir)

        if cfg_path_source != None:  # backup the config file in the log folder
            if cfg_path_source != self.cfg_path:
                copyfile(cfg_path_source, self.cfg_path)

        print("\n[Phase 1] : Data Preparation")
        self.trainer.set_data_loaders(
            self.trainloader, self.testloader, self.mini_testloader
        )

        print("\n[Phase 2] : Model setup")

        print(self.trainer.net)
        print(
            "total # of parameters = {:,} ({:,} trainable)".format(
                sum([torch.numel(p) for p in self.trainer.net.parameters()]),
                sum(
                    [
                        torch.numel(p)
                        for p in filter(
                            lambda x: x.requires_grad, self.trainer.net.parameters()
                        )
                    ]
                ),
            )
        )

        if torch.cuda.is_available():
            self.trainer.net.cuda()
        # Test model: Temporary hack
        self.trainer.test_run_model()

        if resume:
            print("| Resuming from checkpoint...")
            self.load_state()

        print("Initial Validation...")
        acc_valid = self.trainer.run(self.epoch, self.num_epochs, is_training=False)
        # import pdb; pdb.set_trace()
        acc_valid.summarize()
        self.best_metrics = acc_valid
        print("\nSaving the Best Checkpoint...")
        # import pdb; pdb.set_trace()
        self.save_state()
        print(
            "Best Metrics: {acc}\n".format(
                acc=self.best_metrics.summary_str(dtype="scalar", level=0)
            )
        )

        if not test_only:
            print("\n[Phase 3] : Training model")
            print("| Training Epochs = " + str(self.num_epochs))
            elapsed_time = 0
            while self.epoch <= self.num_epochs:
                print("Running at [{}] ...".format(self.exp_dir))
                start_time = time.time()
                acc_train = self.trainer.run(
                    self.epoch, self.num_epochs, is_training=True
                )
                acc_valid = self.trainer.run(
                    self.epoch, self.num_epochs, is_training=False
                )
                acc_mini_test = self.trainer.run(
                    self.epoch, self.num_epochs, is_training=False, mini_test=True
                )

                acc_train.summarize()
                acc_valid.summarize()
                self.log_acc(acc_train, acc_valid)

                self.epoch += 1
                if self.best_metrics_comparator(self.best_metrics, acc_valid):
                    print("\nSaving the Best Checkpoint...")
                    self.best_metrics = acc_valid
                    self.save_state()
                    print(
                        "Best Metrics: {acc}\n".format(
                            acc=self.best_metrics.summary_str(dtype="scalar", level=0)
                        )
                    )
                print(acc_valid)

                epoch_time = time.time() - start_time
                elapsed_time += epoch_time
                print("| Elapsed time : %d:%02d:%02d" % (get_hms(elapsed_time)))

        else:
            print("\n[Phase 4] : Final Performance")
            print("* Test results : {acc}".format(acc=self.best_metrics))

            print("Restoring the Best Checkpoint...")
            self.load_state()
            # self.best_metrics.summarize()
            record = self.best_metrics.filter(dtype="scalar", op=lambda x: float(x))
            offset = 5

            self.trainer.test_run_model()
            l_constant = l2_lipschitz_constant_checker(self.trainer.net)
            # streamline the module during post_steps
            with Streamline(self.trainer.net, True, False):
                print("Current l_constant = {}".format(l_constant))
                for index, post_step in enumerate(self.post_steps):
                    print("\n[Phase {}] : ".format(index + offset), end="")
                    post_step(
                        self.trainer.net,
                        (self.trainloader, self.testloader, self.mini_testloader),
                        l_constant=l_constant,
                        record=record,
                        device=next(self.trainer.net.parameters()).device,
                    )
            print()
            print(yaml.safe_dump(record))
            print("Saving results into a dictionary...")
            print(self.res_dir)
            with open(self.res_dir, "w") as f:
                yaml.safe_dump(record, f)
            print("finished")

