import torch
import webcolors
from PIL import Image, ImageDraw

from minimal.layout import NodeType, NODE_COLOR

def draw_plan(
    masks, # torch.tensor of size (R, 64, 64) (torch.float32)
    nodes,  # list[int] of length R
    img_size=256
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


def blit_rooms(
    rooms: list,
    sep_mask=None,
    out_size=256
):
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
