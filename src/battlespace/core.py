"""
Core Battlespace environment management.
Main container for all environment layers and systems.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import yaml
from pathlib import Path

from .terrain import TerrainLayer
from .structures import StructureLayer
from .airspace import AirspaceLayer
from .weather import WeatherSystem
from .utils.coordinate_utils import validate_position, world_to_grid


class Battlespace:
    """
    Main battlespace environment container.
    Manages terrain, structures, airspace, and weather systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None):
        """
        Initialize battlespace from configuration.
        
        Args:
            config: Configuration dictionary
            config_file: Path to YAML configuration file
        """
        if config_file:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        
        if config is None:
            config = self._default_config()
        
        # Extract dimensions
        self.config = config['battlespace']
        dims = self.config['dimensions']
        self.width = dims['width']
        self.height = dims['height']
        self.altitude_ceiling = dims['altitude_ceiling']
        self.grid_resolution = dims['grid_resolution']
        
        # Calculate grid dimensions
        self.nx = int(self.width / self.grid_resolution)
        self.ny = int(self.height / self.grid_resolution)
        
        # Initialize layers
        self._initialize_layers()
        
        # Cache for performance
        self._elevation_cache = {}
        self._los_cache = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Return default battlespace configuration."""
        return {
            'battlespace': {
                'dimensions': {
                    'width': 50000,  # 50km
                    'height': 50000,  # 50km
                    'altitude_ceiling': 15000,  # 15km
                    'grid_resolution': 100  # 100m
                },
                'terrain': {
                    'generator': 'perlin',
                    'seed': 42,
                    'parameters': {
                        'octaves': 6,
                        'frequency': 0.0001,
                        'amplitude': 2000,
                        'base_elevation': 100
                    }
                },
                'structures': {
                    'enabled': False
                },
                'weather': {
                    'wind': {
                        'base_vector': [10, 0, 0],
                        'altitude_multiplier': 1.5
                    }
                }
            }
        }
    
    def _initialize_layers(self):
        """Initialize all environment layers."""
        # Terrain layer
        terrain_config = self.config.get('terrain', {})
        self.terrain = TerrainLayer(
            self.width, self.height, self.grid_resolution,
            terrain_config
        )
        
        # Structure layer
        structures_config = self.config.get('structures', {})
        self.structures = StructureLayer(
            self.width, self.height,
            structures_config
        )
        
        # Airspace layer
        airspace_config = self.config.get('airspace', {})
        self.airspace = AirspaceLayer(
            self.width, self.height, self.altitude_ceiling,
            airspace_config
        )
        
        # Weather system
        weather_config = self.config.get('weather', {})
        self.weather = WeatherSystem(
            self.width, self.height, self.altitude_ceiling,
            weather_config
        )
    
    def generate(self, seed: Optional[int] = None):
        """
        Generate all battlespace layers.
        
        Args:
            seed: Random seed for reproducible generation
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Generate terrain
        self.terrain.generate()
        
        # Generate structures if enabled
        if self.config.get('structures', {}).get('enabled', False):
            self.structures.generate(self.terrain)
        
        # Generate weather
        self.weather.generate()
        
        # Clear caches after generation
        self._elevation_cache.clear()
        self._los_cache.clear()
    
    def get_elevation(self, x: float, y: float) -> float:
        """
        Get terrain elevation at position.
        
        Args:
            x: East coordinate (meters)
            y: North coordinate (meters)
            
        Returns:
            Elevation in meters
        """
        # Check cache first
        cache_key = (round(x, 1), round(y, 1))
        if cache_key in self._elevation_cache:
            return self._elevation_cache[cache_key]
        
        elevation = self.terrain.get_elevation_at(x, y)
        
        # Cache result
        self._elevation_cache[cache_key] = elevation
        return elevation
    
    def check_collision(self, position: np.ndarray, radius: float = 1.0) -> bool:
        """
        Check if position collides with terrain or structures.
        
        Args:
            position: [x, y, z] position in meters
            radius: Collision radius in meters
            
        Returns:
            True if collision detected
        """
        x, y, z = position
        
        # Check bounds
        if not self.is_in_bounds(x, y, z):
            return True
        
        # Check terrain collision
        terrain_elevation = self.get_elevation(x, y)
        if z <= terrain_elevation + radius:
            return True
        
        # Check structure collision
        if self.structures.check_collision(position, radius):
            return True
        
        return False
    
    def get_wind(self, position: np.ndarray) -> np.ndarray:
        """
        Get wind vector at position.
        
        Args:
            position: [x, y, z] position in meters
            
        Returns:
            Wind vector [vx, vy, vz] in m/s
        """
        return self.weather.get_wind_at(position)
    
    def is_valid_position(self, position: np.ndarray) -> bool:
        """
        Check if position is valid (in bounds and collision-free).
        
        Args:
            position: [x, y, z] position in meters
            
        Returns:
            True if position is valid
        """
        x, y, z = position
        
        # Check bounds
        if not self.is_in_bounds(x, y, z):
            return False
        
        # Check airspace restrictions
        if not self.airspace.is_position_valid(position):
            return False
        
        # Check collisions
        if self.check_collision(position):
            return False
        
        return True
    
    def is_in_bounds(self, x: float, y: float, z: float) -> bool:
        """
        Check if position is within battlespace bounds.
        
        Args:
            x: East coordinate (meters)
            y: North coordinate (meters)
            z: Altitude (meters)
            
        Returns:
            True if position is in bounds
        """
        return (0 <= x <= self.width and 
                0 <= y <= self.height and 
                0 <= z <= self.altitude_ceiling)
    
    def get_line_of_sight(self, pos1: np.ndarray, pos2: np.ndarray, 
                         sample_distance: float = 100.0) -> bool:
        """
        Check if there is clear line of sight between two positions.
        
        Args:
            pos1: First position [x, y, z]
            pos2: Second position [x, y, z]
            sample_distance: Distance between sample points
            
        Returns:
            True if clear line of sight exists
        """
        # Check cache
        cache_key = (tuple(pos1.round(0)), tuple(pos2.round(0)))
        if cache_key in self._los_cache:
            return self._los_cache[cache_key]
        
        # Calculate number of samples
        distance = np.linalg.norm(pos2 - pos1)
        num_samples = max(2, int(distance / sample_distance))
        
        # Sample points along line
        for i in range(num_samples):
            t = i / (num_samples - 1)
            sample_pos = pos1 + t * (pos2 - pos1)
            
            # Check terrain obstruction
            terrain_height = self.get_elevation(sample_pos[0], sample_pos[1])
            if sample_pos[2] <= terrain_height:
                self._los_cache[cache_key] = False
                return False
            
            # Check structure obstruction
            if self.structures.check_line_intersection(pos1, pos2, sample_pos):
                self._los_cache[cache_key] = False
                return False
        
        self._los_cache[cache_key] = True
        return True
    
    def get_minimum_safe_altitude(self, x: float, y: float, 
                                 radius: float = 1000.0,
                                 safety_margin: float = 100.0) -> float:
        """
        Get minimum safe altitude at position considering nearby terrain.
        
        Args:
            x: East coordinate (meters)
            y: North coordinate (meters)
            radius: Search radius for terrain (meters)
            safety_margin: Safety margin above terrain (meters)
            
        Returns:
            Minimum safe altitude in meters
        """
        return self.terrain.get_minimum_safe_altitude(x, y, radius, safety_margin)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get battlespace information.
        
        Returns:
            Dictionary with battlespace parameters
        """
        return {
            'width': self.width,
            'height': self.height,
            'altitude_ceiling': self.altitude_ceiling,
            'grid_resolution': self.grid_resolution,
            'grid_size': (self.nx, self.ny),
            'terrain_range': (self.terrain.min_elevation, self.terrain.max_elevation),
            'num_structures': len(self.structures.structures),
            'num_no_fly_zones': len(self.airspace.no_fly_zones)
        }
    
    def save(self, filepath: str):
        """Save battlespace state to file."""
        # Implementation for saving generated battlespace
        pass
    
    def load(self, filepath: str):
        """Load battlespace state from file."""
        # Implementation for loading saved battlespace
        pass