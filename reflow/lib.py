def calc_min_area(scale: tuple[int, int]):
    return 4

# ==============================

class InputGraph:
    pass

# ==============================

class RectGraph:
    pass

# ==============================

def create_sep_mask(room_masks: list):
    col_mask = torch.ones_like(room_masks[0])
    sep_mask = torch.zeros_like(room_masks[0])

    for rmask in reversed(room_masks):
        col_mask *= (1 - rmask)
        sep_mask += rmask * conv_mask(col_mask, _sep_kernel)

    return sep_mask

# ==============================

# PROCEDURE generate_plan(input_graph: LG, scale: tuple[int, int])

segmentation_masks = run_model(input_graph)
node_types = input_graph.nodes
R = input_graph.num_rooms

# ----------------

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

# ----------------

room_masks = [] # room_mask(r): 1 = yes r / 0 = no r
for graph in rect_graphs:
    room_masks.append(graph.to_mask())

# ----------------
