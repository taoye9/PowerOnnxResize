import os

import numpy as np
import torch
from PIL import Image


def test_real_image():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(parent_dir, "images/test.png")

    print(f"read input image from {image_path}...")

    image_np = np.array(Image.open(image_path))

    print(f"input image loaded success. Image shape: {image_np.shape}")


def export_onnx_graph_1():
    from .ops.resize1 import Resize
    resize = Resize()
    input_tensor = torch.randn(1, 3, 256, 256)
    print(f"Input tensor size: {input_tensor.shape}")
    # resize(input_tensor, upscale_factor)
    with torch.no_grad():
        torch.onnx.export(resize, (input_tensor, ),
                        "resize1.onnx",
                        opset_version=11,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={"input": [2, 3]}
                        )
    
def export_onnx_graph_2():
    from .ops.resize2 import Resize
    resize = Resize()
    input_tensor = torch.randn(1, 3, 256, 256)
    upscale_factor = torch.tensor([3])
    print(f"Input tensor size: {input_tensor.shape}, upscale_factor: {upscale_factor}")
    # resize(input_tensor, upscale_factor)
    with torch.no_grad():
        torch.onnx.export(resize, (input_tensor, upscale_factor),
                        "resize2.onnx",
                        opset_version=16,
                        input_names=['input', 'upscale_factor'],
                        output_names=['output'],
                        dynamic_axes={"input": [2, 3]}
                        )

def export_onnx_graph_3():
    from .ops.resize3 import Resize
    resize = Resize()
    input_tensor = torch.randn(1, 3, 256, 256)
    scale_factor = torch.tensor([3])
    round_factor = torch.tensor([1024])

    print(f"Input tensor size: {input_tensor.shape}, scale_factor: {scale_factor},  round_factor: { round_factor}")
    # resize(input_tensor, upscale_factor)
    with torch.no_grad():
        torch.onnx.export(resize, (input_tensor, scale_factor, round_factor),
                        "resize3.onnx",
                        opset_version=16,
                        input_names=['input', 'scale_factor', 'round_factor'],
                        output_names=['output'],
                        dynamic_axes={"input": [2, 3]}
                        )
         

def main():
    # export_onnx_graph_1()
    # export_onnx_graph_2()
    export_onnx_graph_3()

if __name__ == "__main__":
    main()
