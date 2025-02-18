from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from PIL import Image

try:
    from IPython.display import display

    in_ipython = True
except ImportError:
    in_ipython = False

from minimal.pretrained import model
from minimal.layout import InputGraph, NodeType
from minimal.imaging import draw_plan


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


def _make_edge_triplets(graph: InputGraph):
    """
    Convert graph edges into a tensor of triplets for model input.

    Each triplet (a, c, b) represents an relation between nodes `a` and `b`,
    where `c` is `1` if the nodes are connected and `-1` otherwise. Note that
    the nodes `a` and `b` are ordered such that `a < b`

    Args:
        graph (InputGraph): Graph containing nodes and edges.

    Returns:
        torch.Tensor: Tensor of edge triplets of shape (E, 3).
    """
    n = len(graph.nodes)
    edges = set(graph.edges)

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


@dataclass
class PlanMasks:
    masks: torch.tensor
    graph: InputGraph

    @classmethod
    def create_from_state(cls, state):
        masks = state["masks"]
        graph_dict = state["graph_dict"]

        graph = InputGraph([], [])
        graph.load_state_dict(graph_dict)

        return cls(masks, graph)

    def state_dict(self):
        return {
            "masks": self.masks,
            "graph_dict": self.graph.state_dict(),
        }

    def render(self, img_size=256):
        return draw_plan(self.masks, self.graph.nodes, img_size)

    def __repr__(self):
        if in_ipython:
            display(self.render())

        return f"<PlanMasks {id(self)}>"


def generate_plan(graph: InputGraph, num_iters: int = 10) -> PlanMasks:
    """
    Generate a floor plan layout

    Args:
        graph (InputGraph): Input graph representing the floor plan.
        num_iters (int): Number of refinement iterations.

    Returns:
        torch.Tensor(dtype=torch.byte): Final predicted masks of
        shape (R, 64, 64) (wrapped in a `PlanMasks`)
    """
    nodes = graph.nodes
    edges = graph.edges

    nodes_enc = F.one_hot(
        torch.tensor(nodes), num_classes=NodeType.NUM_NODE_TYPES
    ).float()
    edges_enc = _make_edge_triplets(graph)

    unique_nodes = sorted(list(set(nodes)))

    # Generate initial mask
    masks = _predict_masks(nodes_enc, edges_enc, prev_masks=None, idx_fixed=[])

    for i in range(num_iters):
        fixed_nodes = unique_nodes[:i]

        idx_fixed = [k for k in range(len(nodes)) if nodes[k] in fixed_nodes]

        # Iteratively improve masks
        masks = _predict_masks(
            nodes_enc, edges_enc, prev_masks=masks, idx_fixed=idx_fixed
        )

    # Threshold masks at 0 and convert to binary 0/1
    masks = (masks > 0).byte()

    # remove overlap. modifies the `masks` in-place
    _remove_overlap(graph.nodes, masks)

    return PlanMasks(masks, graph)


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
