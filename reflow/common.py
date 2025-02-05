import torch
import torch.nn.functional as F

def conv_mask(mask, kernel, threshold_match = None):
    mask = mask.to(torch.int8).unsqueeze(0).unsqueeze(0)
    # kernel is assumed to be in int8
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    f = kernel.shape[-1]
    padding = (f - 1) // 2

    result = F.conv2d(mask, kernel, padding=padding)
    result = result[0, 0, :, :]

    if threshold_match is not None:
        result = (result == threshold_match).byte()

    return result
