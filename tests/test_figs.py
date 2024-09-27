from __future__ import annotations

from pathlib import Path
from swmmanywhere_paper.plots import (
    fig4,
    fig5,
    fig6,
    fig78
)

base_dir = Path(__file__).parent / "test_data"
def test_fig4():
    address_path = base_dir / 'cranbrook_node_1439.1' / 'bbox_1' / 'model_12651' / 'addresses.yml'
    real_dir = base_dir / 'cranbrook_node_1439.1' / 'real'
    fig4.plot_fig4(address_path,real_dir)

def test_fig5():
    fig5.plot_fig5(base_dir)

def test_fig6():
    fig6.plot_fig6(base_dir)

def test_fig78():
    fig78.plot_fig78(base_dir)
