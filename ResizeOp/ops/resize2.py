import torch

from torch.nn.functional import interpolate
class ResizeFunc(torch.autograd.Function):

    @staticmethod
    def symbolic(g, input, scales):
        empty_placeholder = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
        return g.op("Resize",
                    input,
                    empty_placeholder,
                    scales,
                    coordinate_transformation_mode_s="pytorch_half_pixel",
                    cubic_coeff_a_f=-0.75,
                    mode_s='cubic',
                    nearest_mode_s="floor")

    @staticmethod
    def forward(ctx, input, scales):
        return input

resize_func = ResizeFunc.apply

class Resize(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, upscale_factor):
        x = resize_func(x, upscale_factor)
        return x
    