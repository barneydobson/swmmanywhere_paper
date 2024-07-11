import os

import geopandas as gpd
import networkx as nx

from swmmanywhere.graph_utilities import register_graphfcn, BaseGraphFunction
from swmmanywhere import geospatial_utilities as go
from swmmanywhere.metric_utilities import nodes_to_subs
from swmmanywhere.filepaths import FilePaths

@register_graphfcn
class trim_to_real(BaseGraphFunction):
    """Test class for graph functions."""
    def __call__(self,
                 G: nx.Graph,
                 addresses: FilePaths,
                 **kwargs) -> nx.Graph:
        """"""
        real_subs = gpd.read_file(addresses.get_path('real_subcatchments'))
        nodes_joined = nodes_to_subs(G, real_subs)
        G = G.subgraph(nodes_joined.id).copy()
        return G

@register_graphfcn
class trim_to_real_subs(BaseGraphFunction):
    """Test class for graph functions."""
    def __call__(self,
                 G: nx.Graph,
                 addresses: FilePaths,
                 **kwargs) -> nx.Graph:
        """"""
        real_subs = gpd.read_file(addresses.get_path('real_subcatchments'))
        syn_subs = gpd.read_file(addresses.model_paths.subcatchments)
        subs_gdf = syn_subs.clip(real_subs)
        
        # sort multi-geometries
        subs_gdf = subs_gdf[~subs_gdf.geometry.is_empty]
        new_geoms = []
        for idx, row in subs_gdf.copy().iterrows():
            if hasattr(row['geometry'],'geoms'):
                new_geoms.extend([g for g in row['geometry'].geoms])
                subs_gdf.drop(idx, inplace=True)
        subs_gdf = go.attach_unconnected_subareas(subs_gdf, new_geoms)

        # Calculate runoff coefficient (RC)
        if addresses.bbox_paths.building.suffix in ('.geoparquet','.parquet'):
            buildings = gpd.read_parquet(addresses.bbox_paths.building)
        else:
            buildings = gpd.read_file(addresses.bbox_paths.building)
        if addresses.model_paths.streetcover.suffix in ('.geoparquet','.parquet'):
            streetcover = gpd.read_parquet(addresses.model_paths.streetcover)
        else:
            streetcover = gpd.read_file(addresses.model_paths.streetcover)

        subs_rc = go.derive_rc(subs_gdf, buildings, streetcover)

        # Write subs
        # TODO - could just attach subs to nodes where each node has a list of subs
        subs_rc.to_file(addresses.model_paths.subcatchments, driver='GeoJSON')

        # Assign contributing area
        imperv_lookup = subs_rc.set_index('id').impervious_area.to_dict()
        
        # Set node attributes
        nx.set_node_attributes(G, 0.0, 'contributing_area')
        nx.set_node_attributes(G, imperv_lookup, 'contributing_area')

        # Prepare edge attributes
        edge_attributes = {edge: G.nodes[edge[0]]['contributing_area'] 
                           for edge in G.edges}

        # Set edge attributes
        nx.set_edge_attributes(G, edge_attributes, 'contributing_area')
        return G