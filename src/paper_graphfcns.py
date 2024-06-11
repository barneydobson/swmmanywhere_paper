import os

import geopandas as gpd
import networkx as nx

from swmmanywhere.graph_utilities import register_graphfcn, BaseGraphFunction
from swmmanywhere.metric_utilities import nodes_to_subs
from swmmanywhere.parameters import FilePaths

@register_graphfcn
class trim_to_real(BaseGraphFunction):
    """Test class for graph functions."""
    def __call__(self,
                 G: nx.Graph,
                 addresses: FilePaths,
                 **kwargs) -> nx.Graph:
        """"""
        real_subs = gpd.read_file(addresses.real_subcatchments)
        nodes_joined = nodes_to_subs(G, real_subs)
        G = G.subgraph(nodes_joined.id).copy()
        return G
