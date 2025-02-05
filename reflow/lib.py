
# room_mask
# neg_room_mask

class InputGraph:
    pass

class GenerationSettings:
    graph: InputGraph
    scale: tuple[int, int]


# PROCEDURE calc_min_area(scale):
#     MIN_AREA_UNSCALED = 10
#     ...

# PROCEDURE generate_plan(input_graph: LG, scale: tuple[int, int])

segmentation_masks = run_model(input_graph)
R = input_graph.num_rooms

min_area_units = calc_min_area(scale)

rooms = []
for i in range(R):
    room_sgm_mask = segmentation_masks[i]
    room_type = input_graph.node_type(i)

    room = RoomAreas(room_type, i, room_sgm_mask)

    # Remove short rectangles
    #
    # TODO: maybe keep rectangles with degree >= 2 as they
    #       will prove to be a "pathway" between multiple
    #       (potentially) disconnected rooms
    # TODO: store the removed rects ("bad" rects) to maybe
    #       salvage them later on
    room.threshold_rectangles(min_area_units)

    # Keep the largest connected component (by total area)
    # 
    # TODO: store the removed rects ("bad" rects) to maybe
    #       salvage them later on
    room.discard_small_components()

    rooms.append(room)
