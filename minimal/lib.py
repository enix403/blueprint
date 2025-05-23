from pathlib import Path
import json
import random
import time

import torch

from minimal.gen import run_model
from minimal.layout import InputLayout, NodeType, NODE_COLOR, into_layout
from minimal.rooms import calc_min_area, RectGraph, scale_room_masks
from minimal.walls import create_sep_mask, scale_sep_mask, extract_face_walls, all_wall_runs
from minimal.join_solving import select_rooms_to_join
from minimal.doors import create_doors, create_front_doors
from minimal.walls import (
    CC_T,
    CC_R,
    CC_B,
    CC_L,
    CC_TL,
    CC_TR,
    CC_BR,
    CC_BL,
)

from pregen.signature import find_closest_graph, graph_folder_name

# Step 1: Generate basic segmentation masks per room
def gen_segmentation_mask(layout: InputLayout):
    masks = run_model(layout, num_iters=25)
    return masks


# Step 2: Convert segmentation masks into a structured plan
def assemble_plan(layout: InputLayout, masks: torch.tensor, scale: tuple[int, int]):
    R = layout.num_rooms()
    node_types = layout.node_types[:R]
    segmentation_masks = masks[:R]

    # ==============================
    # ===== Extract rooms list =====
    # ==============================

    min_area_units = calc_min_area(scale)

    rect_graphs = []
    for i in range(R):
        room_type = node_types[i]
        room_sgm_mask = segmentation_masks[i]

        graph = RectGraph(room_type, i, room_sgm_mask)

        # Remove short rectangles
        #
        # TODO: maybe keep rectangles with degree >= 2 as they
        #       will prove to be a "pathway" between multiple
        #       (potentially) disconnected rooms
        # TODO: store the removed rects ("bad" rects) to maybe
        #       salvage them later on
        graph.threshold_rectangles(min_area_units)

        # Keep the largest connected component (by total area)
        #
        # TODO: store the removed rects ("bad" rects) to maybe
        #       salvage them later on
        graph.discard_small_components()

        rect_graphs.append(graph)

    rect_graphs.sort(key=lambda g: g.total_area())

    # ==============================
    # ===== Create Basic Masks =====
    # ==============================

    room_masks = []  # room_mask(r): 1 = yes r / 0 = no r
    for graph in rect_graphs:
        room_masks.append(graph.to_mask())

    sep_mask = create_sep_mask(room_masks)

    room_masks = scale_room_masks(room_masks, scale)
    sep_mask = scale_sep_mask(sep_mask, *scale)

    # =============================
    # ======= Process Walls =======
    # =============================

    wall_runs = all_wall_runs([
        ((sep_mask & CC_T) | (sep_mask == CC_TR) | (sep_mask == CC_TL)).bool(),
        ((sep_mask & CC_R) | (sep_mask == CC_TR) | (sep_mask == CC_BR)).bool(),
        ((sep_mask & CC_B) | (sep_mask == CC_BR) | (sep_mask == CC_BL)).bool(),
        ((sep_mask & CC_L) | (sep_mask == CC_TL) | (sep_mask == CC_BL)).bool(),
    ])

    face_walls = extract_face_walls(sep_mask)

    # =============================
    # ======== Place Doors ========
    # =============================

    rooms_to_join = select_rooms_to_join(rect_graphs, layout)

    doors = (create_doors(rooms_to_join, room_masks, face_walls)
            + create_front_doors(face_walls, rect_graphs, room_masks, layout))


    return rect_graphs, wall_runs, doors, sep_mask


def generate_plan(
    node_types,
    edges,
    scale
):
    layout = into_layout(node_types, edges)

    # ======== LIBRARY LOOKUP ========
    closest = find_closest_graph(layout)

    if closest is not None:
        # both graphs have the same node count, and almost always
        # copying over the node_labels will be okay
        layout = InputLayout(
            closest.node_types,
            closest.edges,
            closest.node_labels
        )
        print(repr(layout))
        time.sleep(16)

        folder = Path(__file__).parent.parent / "plibrary" / graph_folder_name(layout)
        meta_path = folder / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        count = meta['count']
        i = random.randint(1, count)
        filepath = folder / f"{i}.pth"
        masks = torch.load(filepath)

    # ======== /LIBRARY LOOKUP ========
    else:
        print(repr(layout))
        masks = gen_segmentation_mask(layout)

    rect_graphs, wall_runs, doors, sep_mask = assemble_plan(layout, masks, scale)

    # Sort the graphs back in the original order
    rect_graphs.sort(key=lambda g: g.room_node_index)

    rooms_encoded = []
    for i, r in enumerate(rect_graphs):
        src_index = layout.node_labels[i]
        room_data = [r.room_type, src_index]
        for _, d in r.rects_graph.nodes(data=True):
            room_data.extend(d['xywh'])

        rooms_encoded.append(room_data)

    plan_dict = {
        'shape': tuple(sep_mask.shape),
        'scale': scale,
        "rooms": rooms_encoded,
        "walls": wall_runs,
        "doors": doors
    }

    return plan_dict
