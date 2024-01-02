import torch

import torch._C._onnx as _C_onnx

from torch.nn.functional import interpolate
class ResizeFunc(torch.autograd.Function):

    @staticmethod
    def symbolic(g, input, scale_factor, round_factor):
        empty_placeholder = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))

        shape = g.op("Shape", input)
        shape_float = g.op("Cast", shape, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        shape_short = g.op("Slice", shape_float, 
                           g.op("Constant", value_t=torch.tensor([2], dtype=torch.int32)),
                            g.op("Constant", value_t=torch.tensor([4], dtype=torch.int32)))
    
        shape_1 = g.op("Div", g.op("Mul", shape_short, scale_factor), round_factor)
        shape_3 = g.op("Ceil", shape_1)
        shape_4 = g.op("Mul", shape_3, round_factor)
        shape_int =  g.op("Cast", shape_4, to_i=_C_onnx.TensorProtoDataType.INT64)

        return g.op("Resize",
                    input,
                    empty_placeholder,
                    empty_placeholder,
                    shape_int,
                    coordinate_transformation_mode_s="pytorch_half_pixel",
                    cubic_coeff_a_f=-0.75,
                    mode_s='cubic',
                    nearest_mode_s="floor")

    @staticmethod
    def forward(ctx, input, scale_factor, round_factor):
        return input

resize_func = ResizeFunc.apply

class Resize(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, scale_factor, round_factor):
        x = resize_func(x, scale_factor, round_factor)
        return x
    