import torch

from minimal.walls import (
    BOUNDARY_TOP,
    BOUNDARY_RIGHT,
    BOUNDARY_BOTTOM,
    BOUNDARY_LEFT
)

def _scale_orientation(
    out_mask,
    orient_mask,
    target_orientation,
    scale_x: int, scale_y: int,
    dup_x: bool, dup_y: bool,
    shift_x: bool, shift_y: bool,
):
    loc_x, loc_y = torch.where(orient_mask & target_orientation > 0)

    loc_x *= scale_x
    loc_y *= scale_y

    loc_x = torch.cat([loc_x + i * int(dup_x) for i in range(scale_x)])
    loc_y = torch.cat([loc_y + i * int(dup_y) for i in range(scale_y)])

    loc_x += (scale_x - 1) * int(shift_x)
    loc_y += (scale_y - 1) * int(shift_y)

    out_mask[loc_x, loc_y] = 1


def scale_walls(orient_mask, scale_x, scale_y):

    scaled_walls_mask = torch.zeros((64*scale_x, 64*scale_y), dtype=torch.uint8)

    _scale_orientation(scaled_walls_mask, orient_mask, BOUNDARY_TOP, scale_x, scale_y, 0, 1, 0, 0)
    _scale_orientation(scaled_walls_mask, orient_mask, BOUNDARY_RIGHT, scale_x, scale_y, 1, 0, 0, 1)
    _scale_orientation(scaled_walls_mask, orient_mask, BOUNDARY_BOTTOM, scale_x, scale_y, 0, 1, 1, 0)
    _scale_orientation(scaled_walls_mask, orient_mask, BOUNDARY_LEFT, scale_x, scale_y, 1, 0, 0, 0)

    return scaled_walls_mask