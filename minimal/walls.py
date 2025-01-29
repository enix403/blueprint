from typing import List, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from minimal.rooms import RoomAreas

def _shift_up(m):
    """Shifts a 2D mask `m` up"""
    top, rest = m[:1, :], m[1:, :]
    return torch.cat([rest, top], dim=0)

def _shift_down(m):
    """Shifts a 2D mask `m` down"""
    rest, bottom = m[:-1, :], m[-1:, :]
    return torch.cat([bottom, rest], dim=0)

def _shift_left(m):
    """Shifts a 2D mask `m` left"""
    first, rest = m[:, :1], m[:, 1:]
    return torch.cat([rest, first], dim=1)

def _shift_right(m):
    """Shifts a 2D mask `m` down"""
    rest, last = m[:, :-1], m[:, -1:]
    return torch.cat([last, rest], dim=1)


def walls_between(room_mask, check_room_mask):
    a = room_mask
    b = check_room_mask

    e_top = _shift_down((_shift_up(a) + b == 2).byte())
    e_bottom = _shift_up((_shift_down(a) + b == 2).byte())

    e_left = _shift_right((_shift_left(a) + b == 2).byte())
    e_right = _shift_left((_shift_right(a) + b == 2).byte())

    return (e_top + e_right + e_left + e_bottom).clamp_max_(1)


def intersect_rooms(rooms: list[RoomAreas]):

    # TODO: need a better metric then total_area
    rooms = sorted(rooms, key=lambda r: r.total_area(), reverse=True)
    masks = [room.to_mask() for room in rooms]

    inner_mask = torch.zeros_like(masks[0])
    inner_walls = torch.zeros_like(masks[0])

    for i in range(len(masks)):
        inner_mask += masks[i]
        for j in range(i + 1, len(masks)):
            inner_walls += walls_between(masks[i], masks[j])

    inner_mask.clamp_max_(1)
    inner_walls.clamp_max_(1)

    outer_mask = (1 - inner_mask).byte()
    outer_walls = walls_between(inner_mask, outer_mask)

    walls_mask = (inner_walls + outer_walls).clamp_max_(1)

    return inner_mask, walls_mask

# -----------------

ftl = torch.tensor([
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0],
    [0,  0, -1,  2,  2],
    [0,  0,  2, -1,  0],
    [0,  0,  2,  0,  0],
], dtype=torch.int8)

ftr = torch.tensor([
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0],
    [2,  2, -1,  0,  0],
    [0, -1,  2,  0,  0],
    [0,  0,  2,  0,  0],
], dtype=torch.int8)

fbr = torch.tensor([
    [0,  0,  2,  0,  0],
    [0, -1,  2,  0,  0],
    [2,  2, -1,  0,  0],
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0],
], dtype=torch.int8)

fbl = torch.tensor([
    [0,  0,  2,  0,  0],
    [0,  0,  2, -1,  0],
    [0,  0, -1,  2,  2],
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0],
], dtype=torch.int8)


# ----------------------

fp1 = torch.tensor([
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0],
    [2,  2, -1,  0,  0],
    [0, -1,  2,  2,  0],
    [0,  0,  0,  0,  0],
], dtype=torch.int8)

fp2 = torch.tensor([
    [0, 0,  0,  0,  0],
    [0, 2,  2, -1,  0],
    [0, 0, -1,  2,  2],
    [0, 0,  0,  0,  0],
    [0, 0,  0,  0,  0],
], dtype=torch.int8)

fp3 = torch.tensor([
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0],
    [0,  0, -1,  2,  2],
    [0,  2,  2, -1,  0],
    [0,  0,  0,  0,  0],
], dtype=torch.int8)

fp4 = torch.tensor([
    [0,  0,  0,  0,  0],
    [0, -1,  2,  2,  0],
    [2,  2, -1,  0,  0],
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0],
], dtype=torch.int8)



fp5 = torch.tensor([
    [0,  0,  0,  0,  0],
    [0,  0,  0,  2,  0],
    [0,  0, -1,  2,  0],
    [0,  0,  2, -1,  0],
    [0,  0,  2,  0,  0],
], dtype=torch.int8)

fp6 = torch.tensor([
    [0,  0,  2,  0,  0],
    [0, -1,  2,  0,  0],
    [0,  2, -1,  0,  0],
    [0,  2,  0,  0,  0],
    [0,  0,  0,  0,  0],
], dtype=torch.int8)

fp7 = torch.tensor([
    [0,  0,  2,  0,  0],
    [0,  0,  2, -1,  0],
    [0,  0, -1,  2,  0],
    [0,  0,  0,  2,  0],
    [0,  0,  0,  0,  0],
], dtype=torch.int8)

fp8 = torch.tensor([
    [0,  0,  0,  0,  0],
    [0,  2,  0,  0,  0],
    [0,  2, -1,  0,  0],
    [0, -1,  2,  0,  0],
    [0,  0,  2,  0,  0],
], dtype=torch.int8)

# ----------------------

def conv_mask(mask, kernel, threshold_match = None):
    mask = mask.to(torch.int8).unsqueeze(0).unsqueeze(0)
    # kernel is assumed to be in int8
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    f = kernel.shape[-1]
    padding = (f - 1) // 2

    result = F.conv2d(mask, kernel, padding=padding)
    result = result[0, 0, :, :]

    if threshold_match is not None:
        result = (result == 8).byte()

    return result


def join_wall_corners(walls_mask, inner_mask):
    initial = walls_mask

    walls_mask = walls_mask.clone()

    for kernel in [ftl, ftr, fbr, fbl]:
        res = conv_mask(walls_mask, kernel, 8)
        walls_mask += res
        walls_mask.clamp_max_(1)

    p_walls = torch.zeros_like(initial)

    for kernel in [fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8]:
        res = res = conv_mask(walls_mask, kernel, 8)
        p_walls += res
        p_walls.clamp_max_(1)

    p_walls = torch.logical_and(p_walls, inner_mask)

    walls_mask += p_walls

    return walls_mask


def find_walls(rooms: list[RoomAreas]):
    inner_mask, walls_mask = intersect_rooms(rooms)

    walls_mask = join_wall_corners(walls_mask, inner_mask)
    walls_mask.clamp_max_(1)

    return walls_mask, inner_mask

