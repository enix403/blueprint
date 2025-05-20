from typing import Optional

import torch
import torch.nn.functional as F

from minimal.pretrained import model
from minimal.layout import NodeType, InputLayout, into_layout_unchecked


def _add_interior_doors(layout: InputLayout):
    n = len(layout.node_types)

    door_nodes = []
    door_edges = []

    for a, b in layout.edges:
        if not NodeType.is_room(a) or not NodeType.is_room(b):
            continue

        door_idx = n + len(door_nodes)
        door_nodes.append(NodeType.INTERIOR_DOOR)

        door_edges.append((a, door_idx))
        door_edges.append((b, door_idx))

    node_types = layout.node_types + door_nodes
    edges = layout.edges + door_edges

    return node_types, edges


def _make_edge_triplets(n, edges):
    """
    Convert graph edges into a tensor of triplets for model input.

    Each triplet (a, c, b) represents an relation between nodes `a` and `b`,
    where `c` is `1` if the nodes are connected and `-1` otherwise. Note that
    the nodes `a` and `b` are ordered such that `a < b`

    Returns:
        torch.Tensor: Tensor of edge triplets of shape (E, 3).
    """
    edges = set(edges)

    triplets: list[(int, int, int)] = []

    for a in range(n):
        for b in range(a + 1, n):
            is_joined = ((a, b) in edges) or ((b, a) in edges)
            triplets.append(
                (
                    a,
                    1 if is_joined else -1,
                    b,
                )
            )

    return torch.tensor(triplets, dtype=torch.long)


# ------------------


def _prepare_fixed_masks(masks: torch.tensor, idx_fixed: list[int]):
    """
    Prepare a tensor of fixed and unfixed node masks and their labels.

    Args:
        masks (torch.Tensor): Tensor of shape (R, 64, 64) representing room masks.
        idx_fixed (list[int]): Indices of fixed nodes.

    Returns:
        torch.Tensor: Tensor of shape (R, 2, 64, 64)
    """

    num_nodes = masks.shape[0]

    # (R, 64, 64)
    label_bg = torch.zeros_like(masks)
    masks = masks.clone()

    idx_not_fixed = ([k for k in range(num_nodes) if k not in idx_fixed],)

    # Label the fixed nodes
    label_bg[idx_fixed] = 1.0

    # Label the unfixed nodes, as well as clear
    # out their mask
    label_bg[idx_not_fixed] = 0.0
    masks[idx_not_fixed] = -1.0

    return torch.stack([masks, label_bg], dim=1)


@torch.no_grad()
def _predict_masks(
    nodes_enc: torch.tensor,
    edges_enc: torch.tensor,
    prev_masks: Optional[torch.tensor] = None,
    idx_fixed: list[int] = [],
):
    """
    Predict the next set of masks given node and edge encodings.

    Args:
        nodes_enc (torch.Tensor): Encoded node features of shape (R, NUM_NODE_TYPES).
        edges_enc (torch.Tensor): Encoded edge features of shape (E, 3).
        prev_masks (Optional[torch.Tensor]): Previous masks of shape (R, 64, 64).
        idx_fixed (list[int]): Indices of fixed nodes.

    Returns:
        torch.Tensor: Predicted masks of shape (R, 64, 64).
    """
    num_nodes = nodes_enc.shape[0]

    z = torch.randn(num_nodes, 128)

    if prev_masks is None:
        prev_masks = torch.zeros((num_nodes, 64, 64)) - 1.0

    # (R, 2, 64, 64)
    fixed_masks = _prepare_fixed_masks(prev_masks, idx_fixed)

    next_masks = model(z, fixed_masks, nodes_enc, edges_enc)
    return next_masks.detach()


# ------------------


def _remove_overlap(nodes: list[int], masks: torch.tensor):
    rooms = [(i, node) for i, node in enumerate(nodes) if NodeType.is_room(node)]

    # sort from least important to most important
    rooms.sort(key=lambda room: room[1], reverse=True)

    taken_mask = torch.zeros_like(masks[0])

    for i, _ in rooms:
        # clear room's mask based on taken_mask
        masks[i] = torch.logical_and(masks[i], torch.logical_not(taken_mask))

        # update taken_mask from room's mask
        taken_mask = torch.logical_or(taken_mask, masks[i])


# ------------------
# ------------------


def run_model(
    layout: InputLayout, num_iters: int = 10
) -> torch.tensor:  # shape = (R, 64, 64)
    """
    Generate a new set of floor plan segmentation masks from
    an input layout
    """

    # number of rooms + 1 (for frontdoor)
    R = len(layout.node_types)

    nodes, edges = _add_interior_doors(layout)
    n = len(nodes)

    nodes_enc = F.one_hot(
        torch.tensor(nodes), num_classes=NodeType.NUM_NODE_TYPES
    ).float()
    edges_enc = _make_edge_triplets(n, edges)

    unique_nodes = sorted(list(set(nodes)))

    # Generate initial mask
    masks = _predict_masks(nodes_enc, edges_enc, prev_masks=None, idx_fixed=[])

    num_iters = min(num_iters, len(unique_nodes))

    for i in range(num_iters):
        fixed_nodes = unique_nodes[:i]

        idx_fixed = [k for k in range(n) if nodes[k] in fixed_nodes]

        # Iteratively improve masks
        masks = _predict_masks(
            nodes_enc, edges_enc, prev_masks=masks, idx_fixed=idx_fixed
        )

    # discard doors
    masks = masks[:R]
    nodes = nodes[:R]

    # Threshold masks at 0 and convert to binary 0/1
    masks = (masks > 0).byte()

    # remove overlap. modifies the `masks` in-place
    _remove_overlap(nodes, masks)

    return masks


# ------------


def _create_segmentation_dict(layout, masks):
    return {
        "layout": {"node_types": layout.node_types, "edges": layout.edges},
        "masks": masks,
    }


def _load_segmentation_dict(state):
    masks = state["masks"]

    layout_dict = state["layout"]
    node_types = layout_dict["node_types"]
    edges = layout_dict["edges"]
    layout = into_layout_unchecked(node_types, edges)

    return layout, masks
