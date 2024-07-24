import torch
import torch.nn as nn 


def _check_inputs(func, y0):
    shapes = None
    is_tuple = not isinstance(y0, torch.Tensor)
    if is_tuple:
        assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
        shapes = [y0_.shape for y0_ in y0]
        y0 = torch.cat([y0_.reshape(-1) for y0_ in y0])
        func = _TupleFunc(func, shapes)
    return shapes, func, y0


def _flat_to_shape(tensor, length, shapes):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(tensor[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tuple(tensor_list)


class _TupleFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, g, y, t, for_process):
        f = self.base_func(g, _flat_to_shape(y, (), self.shapes), t, for_process)
        return torch.cat([f_.reshape(-1) for f_ in f])
