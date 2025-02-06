import torch

from minimal.gen import run_model
from minimal.layout import InputLayout, NodeType, NODE_COLOR
from minimal.rooms import calc_min_area, RectGraph, scale_room_masks
from minimal.walls import create_sep_mask, scale_sep_mask, extract_face_walls
from minimal.join_solving import select_rooms_to_join
from minimal.doors import create_doors

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
    # ======== Place Doors ========
    # =============================

    face_walls = extract_face_walls(sep_mask)

    rooms_to_join = select_rooms_to_join(rect_graphs, layout)
    doors = create_doors(R, rooms_to_join, room_masks, face_walls)

    return rect_graphs, sep_mask, doors
