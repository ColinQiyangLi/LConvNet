"""
This file includes a few implemenations of empirical attacks under L2 threat model.
"""
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import grad

import foolbox

def cw_loss(logits, y, c=50, y_target=None, ranked=False):
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, y.view(-1, 1), 1)
    correct_logit = (logits * one_hot).sum(1)
    if y_target is None:
        worst_wrong_logit = logits[one_hot == 0].view(one_hot.size(0), -1).max(1)[0]

        adv_margin = correct_logit - worst_wrong_logit
    elif ranked:
        logit = torch.sort(
            logits[one_hot == 0].view(one_hot.size(0), -1), descending=True
        )[0][:, y_target]
        adv_margin = correct_logit - logit
    else:
        adv_margin = correct_logit - logits[:, y_target]
    return -F.relu(adv_margin + c).mean()


def lp_ball_projection(delta_x, eps, p="inf"):
    if p == "inf":
        return delta_x.clamp(-eps, eps)
    assert p >= 1, "l{} is not a norm".format(d)
    flatten = delta_x.flatten(start_dim=1)
    norm = flatten.norm(p=p, dim=-1, keepdim=True).clamp(min=eps)
    delta_x = (flatten / norm * eps).view(*delta_x.shape)
    return delta_x


def steepest_descent_projection(delta_x, alpha, p="inf"):
    assert p in [
        "inf",
        2,
    ], "only l2 and linf are implemented for steepest descent - {}".format(p)
    if p == "inf":
        return alpha * delta_x.sign()
    if p == 2:
        # return alpha * normalize_by_pnorm(delta_x)
        flatten = delta_x.flatten(start_dim=1)
        norm = flatten.norm(dim=-1, keepdim=True)
        return alpha * (flatten / torch.max(norm, torch.ones_like(norm) * 1e-6)).view(
            *delta_x.shape
        )


def check_norm_ball(delta_x, eps, p="inf"):
    assert p in [
        "inf",
        2,
    ], "only l2 and linf are implemented for steepest descent - {}".format(p)
    if p == "inf":
        return delta_x.flatten(start_dim=1).max(dim=1) > eps
    if p == 2:
        flatten = delta_x.flatten(start_dim=1)
        norm = flatten.norm(dim=-1)
        return norm > (eps + 1e-6)


FOOLBOXATTACKER = {
    2: {
        "cw": foolbox.attacks.CarliniWagnerL2Attack,
        "bia": foolbox.attacks.L2BasicIterativeAttack,
        "boundary_attack": foolbox.attacks.BoundaryAttack,
        "gaussian_blur": foolbox.attacks.GaussianBlurAttack,
        "pointwise": foolbox.attacks.PointwiseAttack,
        "contrast_reduction": foolbox.attacks.ContrastReductionAttack,
        "additive_gaussian": foolbox.attacks.AdditiveGaussianNoiseAttack,
    },
    "inf": {"boundary_attack": foolbox.attacks.BoundaryAttack},
}

# Wrapper for FoolBox attacks
class FoolBoxAttacker:
    def __init__(
        self,
        attack_mode,
        bounds=(0, 1),
        num_classes=10,
        alpha=0.01,
        iters=40,
        *args,
        **kwargs
    ):
        self.attack_mode = attack_mode
        self.bounds = bounds
        self.num_classes = num_classes
        self.alpha = alpha
        self.iters = iters

        self.kwargs = kwargs

    def attack(self, model, x, y, eps, p):
        assert self.attack_mode in FOOLBOXATTACKER[p]
        fb_attack = FOOLBOXATTACKER[p][self.attack_mode]
        fmodel = foolbox.models.PyTorchModel(
            model, bounds=self.bounds, num_classes=self.num_classes
        )
        attack = fb_attack(fmodel)
        adv_s = []
        for xtw, ytw in zip(x, y):
            xt, yt = map(lambda _: _.detach().cpu().numpy(), (xtw, ytw))
            if self.attack_mode in ["boundary_attack"]:
                adv = attack(xt, yt, log_every_n_steps=100000)
            elif self.attack_mode in ["pointwise"]:
                adv = attack(xt, yt)
            elif self.attack_mode in [
                "gaussian_blur",
                "contrast_reduction",
                "additive_gaussian",
            ]:
                adv = attack(xt, yt)
            elif self.attack_mode == "cw":
                adv = attack(xt, yt, **self.kwargs)
            else:
                adv = attack(
                    xt, yt, epsilon=eps, stepsize=self.alpha, iterations=self.iters
                )
            if adv is None:
                print("Attack Failed!")
                adv_s.append(xtw)
            else:
                adv_s.append(torch.from_numpy(adv).to(device=x.device))
                print("Norm:", float((adv_s[-1] - xtw).norm()))
        x_adv = torch.stack(adv_s, dim=0)
        invalid_pts = check_norm_ball(x_adv - x, eps, p)
        x_adv[invalid_pts] = x[invalid_pts]
        return x_adv

class PGDAttacker:
    """
    PGD attack as described in https://arxiv.org/pdf/1706.06083.pdf.
    A slight modification made is that the random initialization is 
    done uniformly within the L2 norm ball.
    """
    def __init__(
        self,
        loss_fn=cw_loss,
        alpha=None,
        iters=40,
        rand_start=True,
        support_projection=None,
        standard_init=True,
    ):
        self.loss_fn = loss_fn
        self.alpha = alpha  # step size
        self.iters = iters
        self.support_projection = (
            support_projection if support_projection is not None else lambda x: x
        )
        self.delta_projection = lambda x, dx: self.support_projection(x + dx) - x
        self.rand_start = rand_start
        self.standard_init = standard_init
        self.attack_mode = "pgd"

    def attack(self, model, x, y, eps, p):
        if self.alpha is None:
            alpha = eps
        else:
            alpha = self.alpha
        dx = torch.zeros_like(x)
        if self.rand_start:
            if p == "inf" or self.standard_init:
                dx.uniform_(-eps, eps)
            else:  # uniformly sample a point from the norm ball
                dx.normal_()
                flatten = dx.flatten(start_dim=1)
                norm = flatten.norm(p=p, dim=-1, keepdim=True)
                u = torch.zeros_like(norm)
                u.uniform_(0, 1)
                interm_dx = flatten / norm
                interm_dx = interm_dx * (u ** (1 / flatten.size(1)))
                dx = interm_dx.view(*dx.shape)
        dx.data = self.delta_projection(x, dx)
        dx.requires_grad_()
        for _ in range(self.iters):
            x_adv = x + dx
            y_pred = model(x_adv)
            loss = cw_loss(y_pred, y)
            dx.data = dx + steepest_descent_projection(grad(loss, dx)[0], alpha, p)
            dx.data = lp_ball_projection(dx, eps, p)
            dx.data = self.delta_projection(x, dx)
        return (x + dx).data

    def __repr__(self):
        return "PGDAttacker({})".format(
            "alpha={alpha}, n_iters={iters}, rand_start={rand_start}".format(
                **self.__dict__
            )
        )


class FGSMAttacker(PGDAttacker):
    """
    FGSM attack is simply one-step PGD attack - https://arxiv.org/pdf/1412.6572.pdf
    """
    def __init__(self, loss_fn=cw_loss, support_projection=None):
        super().__init__(
            loss_fn=loss_fn,
            iters=1,
            rand_start=False,
            support_projection=support_projection,
        )
        self.attack_mode = "fgsm"

    def __repr__(self):
        return "FGSMAttacker()"
