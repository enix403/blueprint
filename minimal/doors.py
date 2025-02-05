def create_cut_wall_mask(wall_mask, doors: list[tuple]):
    walls = (wall_mask > 0).byte()
    for (x1,y1,l,o) in doors:
        x2, y2 = x1 + 1, y1 + 1
        if o == 'h':
            y2 = y1 + l
        else:
            x2 = x1 + l

        walls[x1:x2, y1:y2] = 0

    return walls