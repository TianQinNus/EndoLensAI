"""Minimal timm 0.4 compatibility surface used by the bundled PVTv2 source.

The original repository imports four helpers from timm. E6 only performs
evaluation, so this local fallback avoids adding an old timm package to the
validated Torch 2.9 Windows environment. If timm is installed, it is used.
"""

import sys
import types


def install_if_missing():
    try:
        import timm  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass

    import torch
    from torch import nn

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_probability = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_probability + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_probability) * random_tensor

    def to_2tuple(value):
        return value if isinstance(value, tuple) else (value, value)

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

    def register_model(value):
        return value

    def _cfg(url="", **kwargs):
        return {"url": url, **kwargs}

    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")
    registry = types.ModuleType("timm.models.registry")
    vision_transformer = types.ModuleType("timm.models.vision_transformer")
    layers.DropPath = DropPath
    layers.to_2tuple = to_2tuple
    layers.trunc_normal_ = trunc_normal_
    registry.register_model = register_model
    vision_transformer._cfg = _cfg
    timm.models = models
    models.layers = layers
    models.registry = registry
    models.vision_transformer = vision_transformer
    sys.modules.update(
        {
            "timm": timm,
            "timm.models": models,
            "timm.models.layers": layers,
            "timm.models.registry": registry,
            "timm.models.vision_transformer": vision_transformer,
        }
    )
    return True
