"""
Base class for Lipschitz constrained components. It needs to implement singular_values
"""
from lconvnet.layers.utils import conv_clip_2_norm_numpy, conv_singular_values_numpy

import torch
import torch.nn as nn

class LipschitzModuleL2:
    def singular_values(self):
        raise NotImplementedError

    def lipschitz_constant(self):
        return self.singular_values().max()

# This function computes the Lipschitz constant upper-bound of the entire model, 
# assuming all its components are LipschitzModuleL2
def l2_lipschitz_constant_checker(model, acc=None):
    def extend_collate_op(x, y):
        return y if x is None else x.extend(y)

    l_constant = 1.0
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, LipschitzModuleL2):
            cnt = cnt + 1
            sv = module.singular_values()
            if sv is not None:
                l2_norm = float(sv.max())
                sv = sv.detach().cpu().numpy()
                l_constant *= l2_norm

                # Log the singular value distribution and Lipschitz constant
                if acc is not None:
                    acc(
                        "sanity_check/{name}-l2-norm".format(name=name), l2_norm, dtype="scalar"
                    )
                    acc(
                        "sanity_check/{name}-sv".format(name=name),
                        sv,
                        dtype="histogram",
                        collate_op=extend_collate_op,
                    )
    if cnt == 0: return None
    # Log the overall Lipschitz constant of the model
    if acc is not None:
        acc("sanity_check/layer-l-constant", l_constant, dtype="scalar")
        return acc
    return l_constant
