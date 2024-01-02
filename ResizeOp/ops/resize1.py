import torch

class Resize(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, upscale_factor = 3):
        x = torch.nn.functional.interpolate(x,
                        scale_factor=3,
                        mode='bicubic',
                        align_corners=False)
        return x
    
