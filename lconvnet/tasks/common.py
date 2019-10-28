import torch
import torch.nn.functional as F

def diff_loss(y1, y2):
    return (y2 - y1).mean()

def get_margin_factor(p):
    if p == "inf":
        return 2.0
    return 2.0 ** ((p - 1) / p)

def multi_margin_loss_eps(y_pred, y, eps, p, l_constant, order=1):
    margin = eps * get_margin_factor(p) * l_constant
    return F.multi_margin_loss(y_pred, y, margin=margin, p=order)


def filter_dict(d, key):
    return {k: v for k, v in d.items() if k != key}

def classification_step(
        model, data, optimizer, criterion, is_training, regularizers=[],
        acc=None, device=None, input_hooks=[]):
    x, y = map(lambda x: x.to(device=device), data)
    for input_hook in input_hooks:
        x = input_hook(model, x, y)
    extra_info = {}
    if is_training:
        y_pred = model(x)
        loss = criterion(y_pred, y)
        for reg in regularizers:
            reg_info = reg(model)
            loss += reg_info["loss"]
            extra_info.update(filter_dict(reg_info, "loss"))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        with torch.no_grad():
            y_pred = model(x)
            loss = criterion(y_pred, y)
            for reg in regularizers:
                reg_info = reg(model)
                loss += reg_info["loss"]
                extra_info.update(filter_dict(reg_info, "loss"))

    y_pred_label = y_pred.argmax(dim=-1)
    total = y.size(0)
    correct = int((y_pred_label == y).sum())
    loss = float(loss)
    if acc is not None:
        acc("accuracy", correct / total, dtype="scalar")
        acc("loss", loss, dtype="scalar")
        for key, value in extra_info.items():
            acc(key, value, dtype="scalar")

def wasserstein_distance_estimation_step(
    model, data, optimizer, criterion, is_training, acc=None, device=None
):
    x1, x2 = map(lambda x: x.to(device=device), data)
    if is_training:
        y1, y2 = model(x1), model(x2)
        assert y1.size(-1) == 1 and y2.size(-1) == 1
        loss = criterion(y1, y2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        with torch.no_grad():
            y1, y2 = model(x1), model(x2)
            loss = criterion(y1, y2)

    loss = float(loss)
    if acc is not None:
        acc("loss", loss, dtype="scalar")
