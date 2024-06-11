"""Prepare subselect cut."""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path, PosixPath

from swmmanywhere import swmmanywhere
from src.subselect import subselect_cut
from src import paper_graphfcns # noqa: E402

os.environ['SWMMANYWHERE_VERBOSE'] = "true"

cuts = ['G60F61Y_G60F390_l1',
            'G62F060_G61F180_l1',
            'G74F150_G74F140_l1',
            'G72F550_G72F010_l1',
            'G72F800_G72F050_l1',
            'G73F000_G72F120_l1',
            'G80F390_G80F380_l1',]
subbasin_streamorder = [7,7,7,7,7,7,7]
base_project = 'bellinge'
base_dir = Path(r'C:\Users\bdobson\Documents\data\swmmanywhere')

base_config = swmmanywhere.load_config(base_dir / base_project / 'bf.yml')
base_config['graphfcn_list'].append('trim_to_real')


hpc_config = deepcopy(base_config)
hpc_address = '/rds/general/user/bdobson/ephemeral/swmmanywhere'
hpc_config['base_dir'] = hpc_address
hpc_config['address_overrides']['precipitation'] = \
    '/rds/general/user/bdobson/home/swmmanywhere_paper/tests/test_data/storm.dat'
hpc_config['api_keys'] = \
    '/rds/general/user/bdobson/home/swmmanywhere_paper/tests/test_data/api_keys.yml'
for cut, so in zip(cuts, subbasin_streamorder):
    subselect_cut(base_dir, base_project, cut, buffer = 1/500)
    cut_dir = base_dir / f'{base_project}_{cut}'
    base_config['project'] = f'{base_project}_{cut}'
    base_config['real']['inp'] = cut_dir / 'real' / 'model.inp'
    base_config['real']['graph'] = cut_dir / 'real' / 'graph.json'
    base_config['real']['subcatchments'] = cut_dir / 'real' / 'subcatchments.geojson'
    base_config['address_overrides']['real_subcatchments'] = cut_dir / 'real' / 'subcatchments.geojson'
    base_config['bbox'] = [10.19166667, 55.26609938, 10.40890062, 55.39618666]
    #with (cut_dir / 'real' / 'real_bbox.json').open('r') as f:
    #    base_config['bbox'] = json.load(f)['bbox']
    
    base_config['parameter_overrides'] = {'subcatchment_derivation' :
                                            {'subbasin_streamorder' : so}}
    inp, metrics = swmmanywhere.swmmanywhere(base_config)
    print(inp)
    print(metrics)
    cut_hpc_dir = hpc_address + f'/{base_project}_{cut}/'
    hpc_config['project'] = f'{base_project}_{cut}'
    hpc_config['real']['inp'] = None
    hpc_config['real']['graph'] = cut_hpc_dir + 'real/graph.json'
    hpc_config['real']['subcatchments'] = cut_hpc_dir + 'real/subcatchments.geojson'
    hpc_config['real']['results'] = cut_hpc_dir + 'real/real_results.parquet'
    hpc_config['bbox'] = base_config['bbox']

    swmmanywhere.save_config(hpc_config, cut_dir / 'config.yml')
