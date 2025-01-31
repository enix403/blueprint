import torch

CC_TL = 1
CC_TR = 2
CC_BR = 4
CC_BL = 8

CC_T = 16
CC_R = 32
CC_B = 64
CC_L = 128

_dirs = [
    (-1, -1, CC_TL),
    (-1,  0, CC_T ),
    (-1,  1, CC_TR ),

    ( 0, -1, CC_L ),
    ( 0,  1, CC_R ),

    ( 1, -1, CC_BL),
    ( 1,  0, CC_B ),
    ( 1,  1, CC_BR),
]


def create_sep_mask(room_mask: torch.tensor):
    len_x, len_y = room_mask.shape

    sep_mask = torch.zeros(room_mask.shape, dtype=torch.uint8)

    for x in range(len_x):
        for y in range(len_y):
            cell_room = room_mask[x, y]

            for dx, dy, bit in _dirs:
                nx = x + dx
                ny = y + dy

                if nx < 0 or nx >= len_x or ny < 0 or ny >= len_y:
                    continue

                n_room = room_mask[nx, ny]

                if cell_room > n_room:
                    sep_mask[x, y] += bit

    return sep_mask


def scale_sep_mask(
    sep_mask: torch.tensor,
    scale_x: int, scale_y: int
):
    len_x, len_y = sep_mask.shape

    scaled_sep_mask = torch.zeros(
        (len_x * scale_x, len_y * scale_y),
        dtype=torch.uint8
    )

    for x in range(len_x):
        for y in range(len_y):
            bx = x * scale_x
            by = y * scale_y

            nx = bx + scale_x
            ny = by + scale_y

            view = scaled_sep_mask[bx:nx, by:ny]
            src = sep_mask[x, y].item()

            view[ 0,  0] += src & CC_TL
            view[ 0, -1] += src & CC_TR
            view[-1, -1] += src & CC_BR
            view[-1,  0] += src & CC_BL

            view[ 0,  :] += src & CC_T
            view[ :, -1] += src & CC_R
            view[-1,  :] += src & CC_B
            view[ :,  0] += src & CC_L

    return scaled_sep_mask