import torch
import torch.nn.functional as F

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

# -----------------------------------


def _conv_mask(mask, kernel, threshold_match = None):
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

f = torch.tensor([
    [CC_TL,  CC_T, CC_TR],
    [CC_L ,     0,  CC_R],
    [CC_BL,  CC_B, CC_BR],
], dtype=torch.int8)

def create_sep_mask_fast(room_mask: torch.tensor):
    room_mask = room_mask + 1
    num_rooms = torch.max(room_mask)

    sep_mask = torch.zeros_like(room_mask)

    for r in range(2, num_rooms + 1):
        potential_boundary = (room_mask < r).byte()
        global_seps = _conv_mask(potential_boundary, f)
        global_seps[room_mask != r] = 0

        sep_mask += global_seps

    return sep_mask


def scale_sep_mask_fast(mask: torch.Tensor, scale_x: int, scale_y: int) -> torch.Tensor:
    len_x, len_y = mask.shape
    new_x, new_y = len_x * scale_x, len_y * scale_y

    # Create the expanded mask (replicates each value into a (scale_x, scale_y) block)
    mask_expanded = mask.repeat_interleave(scale_x, dim=0).repeat_interleave(scale_y, dim=1)

    # Create index grids for the expanded space
    x_idx = torch.arange(new_x).view(-1, 1).expand(new_x, new_y)
    y_idx = torch.arange(new_y).view(1, -1).expand(new_x, new_y)

    # Define boolean masks for each region of the expanded cell
    top_row = (x_idx % scale_x) == 0
    bottom_row = (x_idx % scale_x) == (scale_x - 1)
    left_col = (y_idx % scale_y) == 0
    right_col = (y_idx % scale_y) == (scale_y - 1)

    # Initialize the scaled mask
    scaled_mask = torch.zeros((new_x, new_y), dtype=torch.uint8)

    # Apply bitwise operations using the expanded mask
    scaled_mask[top_row & left_col] |= mask_expanded[top_row & left_col] & CC_TL  # Top-left
    scaled_mask[top_row & right_col] |= mask_expanded[top_row & right_col] & CC_TR  # Top-right
    scaled_mask[bottom_row & right_col] |= mask_expanded[bottom_row & right_col] & CC_BR  # Bottom-right
    scaled_mask[bottom_row & left_col] |= mask_expanded[bottom_row & left_col] & CC_BL  # Bottom-left
    scaled_mask[top_row] |= mask_expanded[top_row] & CC_T  # Top edge
    scaled_mask[right_col] |= mask_expanded[right_col] & CC_R  # Right edge
    scaled_mask[bottom_row] |= mask_expanded[bottom_row] & CC_B  # Bottom edge
    scaled_mask[left_col] |= mask_expanded[left_col] & CC_L  # Left edge

    return scaled_mask
