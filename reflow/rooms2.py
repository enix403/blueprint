def scale_room_masks(room_masks, scale):
    sx, sy = scale

    return [
        mask \
            .repeat_interleave(sx, dim=0) \
            .repeat_interleave(sy, dim=1)
        for mask in room_masks
    ]