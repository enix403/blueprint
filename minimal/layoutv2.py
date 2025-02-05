class InputGraph:

    def __init__(
        self,
        node_types: list[int],
        edges: list[tuple[int, int]]
    ):
        G = nx.Graph()

        G.add_nodes_from([
            (i, { "node_type": n })
            for i, n in enumerate(node_types)
        ])
        G.add_edges_from(edges)

        # remove invalid nodes
        G.remove_nodes_from([
            node
            for node, data in G.nodes(data=True)
            if (
                not data['node_type'] in NODE_NAME
                or data['node_type'] == NodeType.INTERIOR_DOOR
            )
        ])

        # keep the largest component
        G = G.subgraph(max(
            nx.connected_components(G), key=len)).copy()

        if len(G) == 0:
            raise Exception("Empty input graph")

        self.G = G

    def ensure_front_door(self):
        G = self.G

        front_doors = [
            node for
            node, data in G.nodes(data=True)
            if data['node_type'] == NodeType.FRONT_DOOR
        ]

        if len(front_doors) > 1:
            # If more than one front doors are present, then
            # delete the extra ones
            G.remove_nodes_from(front_doors[1:])

        elif len(front_doors) == 0:
            # else if none are present, then add one connected
            # to a room (assuming at least one room is present)

            # We connect the front door to the most "important"
            # room (e.g Living room) among the available nodes.
            # The node IDs are assigned such that the most
            # "important" room gets the lowest id.

            min_node = -1
            min_node_type = -1
            next_node = -1

            for node, data in G.nodes(data=True):
                next_node = max(node, next_node)
                node_type = data['node_type']

                if not NodeType.is_room(node_type):
                    continue

                if min_node == -1 or node_type < min_node_type:
                    min_node = node
                    min_node_type = node_type

            next_node += 1
            G.add_node(next_node, node_type=NodeType.FRONT_DOOR)
            G.add_edge(next_node, min_node)

    def draw(self):
        G = self.G

        nx.draw(
            G,
            nx.kamada_kawai_layout(G),
            node_size=1000,
            node_color=[NODE_COLOR[d['node_type']] for n, d in G.nodes(data=True)],
            with_labels=True,
            labels={n: NODE_NAME[d['node_type']] for n, d in G.nodes(data=True)},
            font_color="black",
            font_weight="bold",
            font_size=14,
            edge_color="#b9c991",
            width=2.0,
        )

    def __repr__(self):
        return f"InputGraph(...)"

# ---------

def user_input(node_types: list[int], edges: list[tuple[int, int]]):
    graph = InputGraph(node_types, edges)

    # Make sure that there is a frontdoor
    graph.ensure_front_door()

    # WITHOUT INTERIOR DOORS (ofc)
    # graph.draw()

    return graph