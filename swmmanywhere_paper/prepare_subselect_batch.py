"""Prepare subselect cut."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

from swmmanywhere import swmmanywhere

from swmmanywhere_paper.subselect import subselect_cut

os.environ['SWMMANYWHERE_VERBOSE'] = "true"
base_project = 'cranbrook'

base_dir = Path(r'C:\Users\bdobson\Documents\data\swmmanywhere')

if base_project == 'bellinge':
    cuts = ['G80F390_G80F380_l1',
        'G74F150_G74F140_l1',
        'G60F61Y_G60F390_l1',
            'G62F060_G61F180_l1',    
            'G72F550_G72F010_l1',
            'G72F800_G72F050_l1',
            'G73F000_G72F120_l1',
            ]
    
elif base_project == 'cranbrook':
    cuts = ['node_1439.1']

base_config = swmmanywhere.load_config(base_dir / base_project / 'bf.yml')
base_config['graphfcn_list'].insert(0,'trim_to_real')
base_config['graphfcn_list'].remove('clip_to_catchments')
base_config['graphfcn_list'].insert(11,'trim_to_real_subs')

hpc_config = deepcopy(base_config)
hpc_address = '/rds/general/user/bdobson/ephemeral/swmmanywhere'
hpc_config['base_dir'] = hpc_address
hpc_config['address_overrides']['precipitation'] = \
    '/rds/general/user/bdobson/home/swmmanywhere_paper/tests/test_data/storm.dat'
del hpc_config['address_overrides']['national_building']
for cut in cuts:
    subselect_cut(base_dir, base_project, cut, buffer = 1/500)
    cut_dir = base_dir / f'{base_project}_{cut}'
    base_config['project'] = f'{base_project}_{cut}'
    base_config['real']['inp'] = cut_dir / 'real' / 'model.inp'
    base_config['real']['graph'] = cut_dir / 'real' / 'graph.json'
    base_config['real']['subcatchments'] = cut_dir / 'real' / 'subcatchments.geojson'
    base_config['address_overrides']['real_subcatchments'] = cut_dir / 'real' / 'subcatchments.geojson'
    #base_config['bbox'] = [10.19166667, 55.26609938, 10.40890062, 55.39618666]
    with (cut_dir / 'real' / 'real_bbox.json').open('r') as f:
        base_config['bbox'] = json.load(f)['bbox']

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
    hpc_config['address_overrides']['real_subcatchments'] = hpc_config['real']['subcatchments']
    

    swmmanywhere.save_config(hpc_config, cut_dir / 'config.yml')