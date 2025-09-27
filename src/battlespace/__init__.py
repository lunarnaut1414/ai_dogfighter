"""
Battlespace environment package for autonomous interceptor simulation.
"""

from .core import Battlespace
from .terrain import TerrainLayer, TerrainType
from .structures import StructureLayer, Structure, StructureType
from .airspace import AirspaceLayer, NoFlyZone, ThreatZone, ControlledAirspace
from .weather import WeatherSystem

__all__ = [
    'Battlespace',
    'TerrainLayer',
    'TerrainType',
    'StructureLayer',
    'Structure',
    'StructureType',
    'AirspaceLayer',
    'NoFlyZone',
    'ThreatZone',
    'ControlledAirspace',
    'WeatherSystem'
]

__version__ = '0.1.0'