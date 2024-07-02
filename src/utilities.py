from typing import Hashable
import pandas as pd
import networkx as nx 

def calc_flowtimes(G: nx.MultiDiGraph, 
                   df: pd.DataFrame,
                   outlet: Hashable,
                   cutoff = pd.to_datetime('2000-01-01 03:00:00')):
    """
    Calculate the flow times for a network.

    Args:
        G (nx.MultiDiGraph): The input graph.
        df (pd.DataFrame): The input DataFrame.
        outlet (Hashable): The outlet node.
        cutoff (pd.Timestamp): The cutoff date. Defaults to '2000-01-01 03:00:00'.
    """
    df = df[df['date'] < cutoff]
    df = df.pivot(columns = 'variable',values='value',index = ['date','id'])
    df['velocity'] = df['flow'] / ((df['ups_xsection_area'] + df['ds_xsection_area']) / 2)
    vel = df.groupby('id').velocity.mean()
    vel[vel < 0.00001] = 0.00001
    vel = vel.to_dict()
    nx.set_edge_attributes(G,
                           {(u,v,k): d['length'] / vel[d['id']] 
                            for u,v,k,d in G.edges(data=True,keys=True)},
                           'travel_time')
    paths = nx.shortest_path(G,
                             target = outlet,
                             weight = 'travel_time')
    times = {n : 0 for n in G.nodes}
    for node, path in paths.items():
        for u,v in zip(path[:-1],path[1:]):
            times[node] += G[u][v][0]['travel_time']
    nx.set_node_attributes(G,
                           times,
                           'travel_time')

    return G