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
