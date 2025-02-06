import torch
import webcolors
from PIL import Image, ImageDraw

from minimal.layout import NodeType, NODE_COLOR
from minimal.walls import CC_TL, CC_TR, CC_BR, CC_BL, CC_T, CC_R, CC_B, CC_L


def draw_plan(
    masks,  # torch.tensor of size (R, 64, 64) (torch.float32)
    nodes,  # list[int] of length R
    img_size=256,
):
    plan_img = Image.new("RGB", (64, 64), (255, 255, 255))
    draw = ImageDraw.Draw(plan_img)

    for m, n in zip(masks, nodes):
        if NodeType.is_door(n):
            continue

        mask_bitmap = Image.fromarray(m.numpy() * 255, mode="L")
        r, g, b = webcolors.hex_to_rgb(NODE_COLOR[n])
        draw.bitmap((0, 0), mask_bitmap, fill=(r, g, b))

    return plan_img.resize((img_size, img_size), Image.Resampling.BOX)


def blit_rooms(rooms: list, sep_mask=None, out_size=256):
    h = rooms[0].grid_height
    w = rooms[0].grid_width

    plan_img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(plan_img)

    for room in rooms:
        m = room.to_mask()
        n = room.room_type

        mask_bitmap = Image.fromarray(m.numpy() * 255, mode="L")
        r, g, b = webcolors.hex_to_rgb(NODE_COLOR[n])
        draw.bitmap((0, 0), mask_bitmap, fill=(r, g, b))

    if sep_mask is not None:
        walls = (sep_mask > 0).byte()
        plan_img = plan_img.resize(tuple(walls.shape), Image.Resampling.BOX)

        mask_bitmap = Image.fromarray(walls.numpy() * 255, mode="L")
        ImageDraw.Draw(plan_img).bitmap((0, 0), mask_bitmap, fill=(0, 0, 0))

    return plan_img.resize((out_size, out_size), Image.Resampling.BOX)


def draw_sep_nask_wireframe(
    mask: torch.Tensor, cell_size: int = 40, dot_radius: int = 4
):
    # Constants
    bg_color = "#282828"
    dot_color = "#8BFF6B"
    edge_line_color = "#FFA66B"
    center_line_color = "#C63735"

    len_x, len_y = mask.shape
    img_size = (len_y * cell_size, len_x * cell_size)

    img = Image.new("RGB", img_size, bg_color)
    draw = ImageDraw.Draw(img)

    # Iterate over the mask positions
    for x in range(len_x):
        for y in range(len_y):
            cell_x = y * cell_size
            cell_y = x * cell_size
            center = (cell_x + cell_size // 2, cell_y + cell_size // 2)

            # Read the bitmask value for this cell
            cell_value = mask[x, y].item()

            # Define positions for dots
            positions = {
                CC_TL: (cell_x, cell_y),  # Top-left
                CC_TR: (cell_x + cell_size, cell_y),  # Top-right
                CC_BR: (cell_x + cell_size, cell_y + cell_size),  # Bottom-right
                CC_BL: (cell_x, cell_y + cell_size),  # Bottom-left
                CC_T: (cell_x + cell_size // 2, cell_y),  # Top edge (middle)
                CC_R: (
                    cell_x + cell_size,
                    cell_y + cell_size // 2,
                ),  # Right edge (middle)
                CC_B: (
                    cell_x + cell_size // 2,
                    cell_y + cell_size,
                ),  # Bottom edge (middle)
                CC_L: (cell_x, cell_y + cell_size // 2),  # Left edge (middle)
            }

            # Draw edge connections between corners first (so dots render on top)
            if cell_value & CC_T:  # Top edge
                draw.line(
                    [positions[CC_TL], positions[CC_TR]], fill=edge_line_color, width=2
                )
            if cell_value & CC_R:  # Right edge
                draw.line(
                    [positions[CC_TR], positions[CC_BR]], fill=edge_line_color, width=2
                )
            if cell_value & CC_B:  # Bottom edge
                draw.line(
                    [positions[CC_BL], positions[CC_BR]], fill=edge_line_color, width=2
                )
            if cell_value & CC_L:  # Left edge
                draw.line(
                    [positions[CC_TL], positions[CC_BL]], fill=edge_line_color, width=2
                )

            # Draw dots and lines to center
            for bit, pos in positions.items():
                if cell_value & bit:
                    draw.line([center, pos], fill=center_line_color, width=2)
                    draw.ellipse(
                        (
                            pos[0] - dot_radius,
                            pos[1] - dot_radius,
                            pos[0] + dot_radius,
                            pos[1] + dot_radius,
                        ),
                        fill=dot_color,
                        # outline="white"
                    )

    return img
