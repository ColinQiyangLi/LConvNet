import torch
from lconvnet.utils import Accumulator, Streamline

import sys

GLOBAL_DICT = {}

class StepScheduler:
    def __init__(self, init_v, min_v=None, max_v=None, increase_x_every_y=None):
        self.init_v = init_v
        self.max_v = max_v
        self.min_v = min_v
        self.v = init_v
        self.x, self.y = increase_x_every_y

    def step(self, epoch):
        self.v = self.init_v + (epoch // self.y) * self.x
        if self.max_v is not None:
            self.v = min(self.v, self.max_v)
        if self.min_v is not None:
            self.v = max(self.v, self.min_v)

    def get_lr(self):
        return self.v


class GlobalScheduler:
    def __init__(self, scheduler, key="epoch"):
        self.scheduler = scheduler
        self.key = key

    def update(self):
        global GLOBAL_DICT
        self.scheduler.step(GLOBAL_DICT[self.key])

    def get(self):
        return self.scheduler.get_lr()

class Trainer:
    def __init__(
            self, net, optimizer, lr_scheduler, main_step, post_steps=[],
            proj_every_n_its=20, corrupted_label=None):
        self.net = net
        self.optimizer = optimizer(self.net.parameters())
        self.lr_scheduler = lr_scheduler(self.optimizer)

        self.main_step = main_step
        self.post_steps = post_steps

        self.proj_every_n_its = proj_every_n_its
        self.corrupted_label = corrupted_label

        self.trainloader, self.validloader, self.testloader = None, None, None
        if torch.cuda.is_available():
            # torch.cuda.device(torch.cuda.current_device())
            self.device = "cuda"
        else:
            assert False, "There is no cuda device available!!"
            self.device = "cpu"
        global GLOBAL_DICT
        GLOBAL_DICT["epoch"] = 0

    def state_dict(self):
        return {"net": self.net.state_dict()}

    def load_state_dict(self, d):
        self.net.load_state_dict(d["net"])

    def set_data_loaders(self, trainloader, testloader, mini_testloader=None):
        self.trainloader = trainloader
        self.testloader = testloader
        self.mini_testloader = mini_testloader

    def test_run_model(self):
        for data in self.testloader:
            self.main_step(self.net, data, self.optimizer,
                           is_training=False, device=self.device)
            break

    def run(self, epoch, num_epochs, is_training=True, mini_test=False):
        self.lr_scheduler.step(epoch)
        global GLOBAL_DICT
        GLOBAL_DICT["epoch"] = epoch
        assert (
            self.trainloader is not None and self.testloader is not None
        ), "the data loaders has not been set"
        if is_training:
            self.net.train()
            print("\n=> Training Epoch #{:d}, LR={:.4f}, GLOBAL_DICT={}".format(
                epoch, self.lr_scheduler.get_lr()[0], GLOBAL_DICT))
            its = self.trainloader
        else:
            self.net.eval()
            print("\n=> Validation Epoch #%d" % (epoch))
            if mini_test:
                its = self.mini_testloader
            else:
                its = self.testloader

        length = len(its)

        acc = Accumulator()
        old_flag, new_flag = (False, False) if is_training else (False, True)

        # streamline the module during validation
        with Streamline(self.net, new_flag, old_flag):
            for batch_idx, data in enumerate(its):
                if self.corrupted_label is not None:
                    x, y = data
                    bs = len(y)
                    cbs = int(bs * self.corrupted_label)
                    y[:cbs] = (y[:cbs] + torch.randint(1, 10, cbs)
                               ) % 10  # HARD-CODED
                    data = (x, y)
                self.main_step(self.net, data, self.optimizer,
                               is_training=is_training, acc=acc, device=self.device)
                if is_training:
                    if batch_idx == len(its) - 1 or batch_idx % self.proj_every_n_its == 0:
                        for module in self.net.modules():
                            if hasattr(module, "_project"):
                                module._project()
                sys.stdout.write("\r | Epoch [{:03d}/{:03d}] Iter[{:03d}/{:03d}] | {acc}".format(
                    epoch, num_epochs, batch_idx + 1, length, acc=acc.latest_str()))
                sys.stdout.flush()
        for post_step in self.post_steps:
            post_step(self.net, acc)

        return acc
