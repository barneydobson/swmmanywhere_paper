# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, model_validator


class SubcatchmentDerivation(BaseModel):
    """Parameters for subcatchment derivation."""
    lane_width: float = Field(default = 3.5,
            ge = 2.0,
            le = 5.0,
            unit = "m", 
            description = "Width of a road lane.")

    carve_depth: float = Field(default = 2.0,
            ge = 1.0,
            le = 3.0,
            unit = "m", 
            description = "Depth of road/river carve for flow accumulation.")

    max_street_length: float = Field(default = 60.0,
            ge = 20.0,
            le = 100.0,
            unit = "m", 
            description = "Distance to split streets into segments.")

class OutletDerivation(BaseModel):
	"""Parameters for outlet derivation."""
	max_river_length: float = Field(default = 30.0,
		ge = 5.0,
		le = 100.0,
		unit = "m",
		description = "Distance to split rivers into segments.")   

	river_buffer_distance: float = Field(default = 150.0,
		ge = 50.0,
		le = 300.0,
		unit = "m",
		description = "Buffer distance to link rivers to streets.")

	outlet_length: float = Field(default = 40.0,
		ge = 10.0,
		le = 600.0,
		unit = "m",
		description = "Length to discourage street drainage into river buffers.")

class TopologyDerivation(BaseModel):
    """Parameters for topology derivation."""
    weights: list = Field(default = ['surface_slope',
                                      'chahinan_angle',
                                      'length',
                                      'contributing_area'],
                        min_items = 1,
                        unit = "-",
                        description = "Weights for topo derivation")

    surface_slope_scaling: float = Field(default = 1,
        le = 1,
        ge = 0,
        unit = "-",
        description = "Constant to apply to surface slope in topo derivation")
    
    chahinan_angle_scaling: float = Field(default = 1,
        le = 1,
        ge = 0,
        unit = "-",
        description = "Constant to apply to chahinan angle in topo derivation")
    
    length_scaling: float = Field(default = 1,
        le = 1,
        ge = 0,
        unit = "-",
        description = "Constant to apply to length in topo derivation")
    
    contributing_area_scaling: float = Field(default = 1,
        le = 1,
        ge = 0,
        unit = "-",
        description = "Constant to apply to contributing area in topo derivation")
    
    surface_slope_exponent: float = Field(default = 1,
        le = 2,
        ge = -2,
        unit = "-",
        description = "Exponent to apply to surface slope in topo derivation")
    
    chahinan_angle_exponent: float = Field(default = 1,
        le = 2,
        ge = -2,
        unit = "-",
        description = "Exponent to apply to chahinan angle in topo derivation")

    length_exponent: float = Field(default = 1,
        le = 2,
        ge = -2,
        unit = "-",
        description = "Exponent to apply to length in topo derivation")
    
    contributing_area_exponent: float = Field(default = 1,
        le = 2,
        ge = -2,
        unit = "-",
        description = "Exponent to apply to contributing area in topo derivation")
    

    
    @model_validator(mode='after')
    def check_weights(cls, values):
        """Check that weights have associated scaling and exponents."""
        for weight in values.weights:
                if not hasattr(values, f'{weight}_scaling'):
                        raise ValueError(f"Missing {weight}_scaling")
                if not hasattr(values, f'{weight}_exponent'):
                        raise ValueError(f"Missing {weight}_exponent")
        return values

# TODO move this to tests and run it if we're happy with this way of doing things
class NewTopo(TopologyDerivation):
     """Demo for changing weights that should break the validator."""
     weights: list = Field(default = ['surface_slope',
                                      'chahinan_angle',
                                      'length',
                                      'contributing_area',
                                'test'],
                        min_items = 1,
                        unit = "-",
                        description = "Weights for topo derivation")

class HydraulicDesign(BaseModel):
    """Parameters for hydraulic design."""
    diameters: list = Field(default = np.linspace(0.15,3,int((3-0.15)/0.075) + 1),
                            min_items = 1,
                            unit = "m",
                            description = """Diameters to consider in 
                            pipe by pipe method""")
    max_fr: float = Field(default = 0.8,
		upper_limit = 1,
		lower_limit = 0,
		unit = "-",
		description = "Maximum filling ratio in pipe by pipe method")
    min_shear: float = Field(default = 2,
		upper_limit = 3,
		lower_limit = 0,
		unit = "Pa",
		description = "Minimum wall shear stress in pipe by pipe method")
    min_v: float = Field(default = 0.75,
		upper_limit = 2,
		lower_limit = 0,
		unit = "m/s",
		description = "Minimum velocity in pipe by pipe method")
    max_v: float = Field(default = 5,
		upper_limit = 10,
		lower_limit = 3,
		unit = "m/s",
		description = "Maximum velocity in pipe by pipe method")
    min_depth: float = Field(default = 0.5,
		upper_limit = 1,
		lower_limit = 0,
		unit = "m",
		description = "Minimum excavation depth in pipe by pipe method")
    max_depth: float = Field(default = 5,
		upper_limit = 10,
		lower_limit = 2,
		unit = "m",
		description = "Maximum excavation depth in pipe by pipe method")
    precipitation: float = Field(default = 0.006,
		upper_limit = 0.010,
		lower_limit = 0.001,
		description = "Depth of design storm in pipe by pipe method",
		unit = "m")

class FilePaths:
    """Parameters for file path lookup.

    TODO: this doesn't validate file paths to allow for un-initialised data
    (e.g., subcatchments are created by a graph and so cannot be validated).
    """

    def __init__(self, 
                 base_dir: Path, 
                 project_name: str, 
                 bbox_number: int, 
                 model_number: str, 
                 extension: str='json'):
        """Initialise the class."""
        self.base_dir = base_dir
        self.project_name = project_name
        self.bbox_number = bbox_number
        self.model_number = model_number
        self.extension = extension
    
    def __getattr__(self, name):
        """Fetch the address."""
        return self._fetch_address(name)
    
    def _generate_path(self, *subdirs):
        """Generate a path."""
        return self.base_dir.joinpath(*subdirs)

    def _generate_property(self, 
                           property_name: str, 
                           location: str):
        """Generate a property.
        
        Check if the property exists in the class, otherwise generate it.
        
        Args:
            property_name (str): Name of the folder/file.
            location (str): Name of the folder that the property_name exists 
                in.
            
        Returns:
            Path: Path to the property.
        """
        if property_name in self.__dict__.keys():
             return self.__dict__[property_name]
        else:
            return self._generate_path(self.project_name, 
                                       getattr(self, location),
                                       property_name)

    def _fetch_address(self, name):
        """Fetch the address.
        
        Generate a path to the folder/file described by name. If the 
        folder/file has already been set, then it will be returned. Otherwise
        it will be generated according to the default structure defined below.

        Args:
            name (str): Name of the folder/file.
            
        Returns:
            Path: Path to the folder/file.
        """
        if name == 'project':
            return self._generate_path(self.project_name)
        elif name == 'national':
             return self._generate_property('national', 'project')
        elif name == 'bbox':
            return self._generate_property(f'bbox_{self.bbox_number}', 
                                       'project')
        elif name == 'model':
            return self._generate_property(f'model_{self.model_number}', 
                                       'bbox')
        elif name == 'subcatchments':
            return self._generate_property(f'subcatchments.{self.extension}', 
                                       'model')
        elif name == 'download':
            return self._generate_property('download', 
                                            'bbox')
        elif name == 'elevation':
            return self._generate_property('elevation.tif', 'download')
        elif name == 'building':
            return self._generate_property(f'building.{self.extension}', 
                                       'download')
        elif name == 'precipitation':
            return self._generate_property(f'precipitation.{self.extension}', 
                                       'download')
        else:
            raise AttributeError(f"Attribute {name} not found")
    