from minimal.correction import RoomAreas

def is_adjacent(rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Check if they are adjacent horizontally or vertically
        if x1 + h1 == x2 or x2 + h2 == x1:  # Vertical adjacency
            return not (y1 + w1 <= y2 or y2 + w2 <= y1)
        if y1 + w1 == y2 or y2 + w2 == y1:  # Horizontal adjacency
            return not (x1 + h1 <= x2 or x2 + h2 <= x1)
        return False

def spacial_adjacency_graph(rooms: list[RoomAreas]):
    # Constructs a graph where:
    #   - Each node is a room (RoomAreas i.e set of rects)
    #   - Each edge represents spatial adjacency between rooms
    #   - Each edge also contains the "touching" portion of the
    #     boundary of the adjacent rooms

    pass