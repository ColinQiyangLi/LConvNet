"""
Evaluate adversarial robustness of a given classifier.
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict

from lconvnet.utils import Accumulator


def accuracy_natural(model, x, y, avg=True):
    preds = model(x).detach().argmax(dim=-1)
    if avg:
        return float((preds == y).float().mean())
    return preds == y


def pnorm(x, p):
    x = x.flatten(start_dim=1)
    assert p in ["inf", 2]
    if p == "inf":
        return x.max(dim=-1)
    if p == 2:
        return x.norm(dim=-1)


def accuracy_upperbound(model, attacker, x, y, eps, p, avg=True, returns_extra=False):
    """
    Wrapper function that calls the attacker to perturb the input example against the model.
    This computes an upper-bound on the robust accuracy of a model
    """
    x_adv = attacker.attack(model, x, y, eps=eps, p=p)
    success = model(x_adv).detach().argmax(dim=-1) != y
    in_bound = pnorm(x_adv - x, p) < eps + 1e-6  # add a numerical tolerance

    success_in_bound = success & in_bound
    acc = ~success_in_bound
    dist = (x_adv[success] - x[success]).flatten(start_dim=1).norm(dim=-1)
    if avg:
        acc, dist, success = map(
            lambda x: float(x.float().mean()), (acc, dist, success)
        )
    if returns_extra:
        return acc, dist, success
    return acc


def accuracy_lowerbound_lnet(
    model, x, y, eps, l_constant, p, avg=True, returns_extra=False
):
    """
    Compute the lower-bound robust accuracy of a model with known Lipschitz constant upper-bound
    """
    logits = model(x).detach()
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, y.view(-1, 1), 1)
    correct_logit = (logits * one_hot).sum(1)
    worst_wrong_logit = logits[one_hot == 0].view(one_hot.size(0), -1).max(1)[0]
    adv_margin = F.relu(correct_logit - worst_wrong_logit)
    if p == "inf":
        fact = 2.0
    else:
        fact = 2.0 ** ((p - 1) / p)

    def op(eps_t):
        acc = adv_margin > fact * eps_t * l_constant
        dist = adv_margin / fact / l_constant
        if avg:
            acc, dist = map(lambda x: float(x.float().mean()), (acc, dist))
        if returns_extra:
            return acc, dist
        return acc

    if type(eps) == list:
        return list(map(op, eps))
    return op(eps)


def eval_adv_robustness_batch(
    model,
    data_loaders,
    attacker=None,
    eps_range=None,
    p=None,
    l_constant=None,
    device=None,
    record=None,
    mini_eval=False,
    force_lipschitz_constant=None,
    verbose=True,
):
    """
    Evaluate adversarial robustness of a given model on a test data stream (specified by a tuple of data_loaders).
    The l_constant/force_lipschitz_constant will be used for determining the lower-bound on the robust accuracy given
    the eps of interest (specified by eps_range).
    """
    train_dataloader, test_dataloader, mini_test_dataloader = data_loaders
    assert eps_range is not None
    if force_lipschitz_constant is not None:
        l_constant = force_lipschitz_constant

    if verbose:
        params_for_display = {
            "attacker": attacker,
            "eps_range": eps_range if eps_range is not None else "None",
            "p": p if p is not None else "None",
            "l_constant": l_constant if l_constant is not None else "None",
        }
        print(
            "Evaluating Adversarial Robustness in Batch: \n| {}".format(
                "\n| ".join(
                    [
                        "{} : {}".format(key, value)
                        for key, value in params_for_display.items()
                    ]
                )
            )
        )

    if mini_eval:
        print("Using mimi test data...")
    else:
        print("Using whole test data...")

    summary = {}
    for eps in eps_range:
        print("\neps={:2f}".format(eps))
        res = eval_adv_robustness(
            model,
            mini_test_dataloader if mini_eval else test_dataloader,
            attacker=attacker,
            eps=float(eps),
            p=p,
            l_constant=l_constant,
            device=device,
            verbose=False,
        )
        summary[eps] = res

    for eps in sorted(summary):
        res = summary[eps]
        if "lb" in res:
            ub_repr = " ub: {:.2f}% |".format(100.0, 100.0 - res["lb"] * 100.0)
        else:
            ub_repr = ""
        if "ub" in res:
            lb_repr = " lb: {:.2f}% |".format(100.0, 100.0 - res["ub"] * 100.0)
        else:
            lb_repr = ""

        print(
            "{:.2f} | clean: {:.2f}% |{}{}".format(
                eps, 100.0 - res["natural"] * 100.0, ub_repr, lb_repr
            )
        )

    if record is not None:
        if mini_eval:
            if "adv_robustness_mini_eval" not in record:
                record["adv_robustness_mini_eval"] = {}
            record["adv_robustness_mini_eval"].update(summary)
        else:
            if "adv_robustness" not in record:
                record["adv_robustness"] = {}
            record["adv_robustness"].update(summary)


def eval_adv_robustness(
    model,
    data_loader,
    attacker=None,
    eps=None,
    p=None,
    l_constant=None,
    device=None,
    verbose=True,
):

    print("l-constant:", l_constant)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        params_for_display = {
            "attacker": attacker,
            "eps": eps if eps is not None else "None",
            "p": p if p is not None else "None",
            "l_constant": l_constant if l_constant is not None else "None",
        }
        print(
            "Evaluating Adversarial Robustness: \n| {}".format(
                "\n| ".join(
                    [
                        "{} : {}".format(key, value)
                        for key, value in params_for_display.items()
                    ]
                )
            )
        )

    acc = Accumulator()
    total = len(data_loader)
    if attacker is not None and type(attacker) != list:
        attacker = [attacker]
    for index, (x, y) in enumerate(data_loader):
        if device is not None:
            x = x.to(device=device)
            y = y.to(device=device)
        if l_constant is not None:
            lb, lb_dist = accuracy_lowerbound_lnet(
                model, x, y, eps, l_constant, p, avg=False, returns_extra=True
            )
            acc("lb", float(lb.float().mean()), dtype="scalar")
            acc("lb_dist", list(map(float, lb_dist.float())), dtype="scalar")
        else:
            lb = None
        if attacker is not None:
            for att in attacker:
                ub, ub_dist, ub_success = accuracy_upperbound(
                    model, att, x, y, eps, p, avg=False, returns_extra=True
                )

                # Check if the emperical attacks are successful at a certified point
                if lb is not None and l_constant is not None:
                    assert (
                        (lb & ub) | (~lb)
                    ).all(), "robustness sanity check failed at eps={}".format(eps)

                # Record the upperbound statistics
                acc(
                    "{att}-ub".format(att=att.attack_mode),
                    float(ub.float().mean()),
                    dtype="scalar",
                )
                if ub_dist is not None:
                    acc(
                        "{att}-ub-dist".format(att=att.attack_mode),
                        list(map(float, ub_dist.float())),
                        dtype="scalar",
                    )
                acc(
                    "{att}-ub-success-rate".format(att=att.attack_mode),
                    float(ub_success.float().mean()),
                    dtype="scalar",
                )
        acc("natural", accuracy_natural(model, x, y), dtype="scalar")
        print("\r[{}/{}] - {}".format(index + 1, total, acc.latest_str()), end="")
    acc.summarize()
    res = acc.collect()

    if verbose:
        if "natural" in res:
            print(
                "\n\nClean Test Error: {:.3f}%".format(100.0 - res["natural"] * 100.0)
            )
        if "lb" in res:
            print("Upper Bound Test Error: {:.3f}%".format(100.0 - res["lb"] * 100.0))
        if "ub" in res:
            print("Lower Bound Test Error: {:.3f}%".format(100.0 - res["ub"] * 100.0))
    return res
