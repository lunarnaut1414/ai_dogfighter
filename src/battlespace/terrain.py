"""
Terrain layer management for battlespace environment.
Handles elevation generation, queries, and terrain analysis.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from enum import IntEnum

from .generators.terrain_generator import TerrainGenerator
from .utils.interpolation import bilinear_interpolate


class TerrainType(IntEnum):
    """Terrain type classifications based on elevation."""
    WATER = 0
    GRASS = 1
    DIRT = 2
    ROCK = 3
    SNOW = 4


class TerrainLayer:
    """
    Manages terrain elevation and properties.
    """
    
    def __init__(self, width: float, height: float, resolution: float,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize terrain layer.
        
        Args:
            width: Width in meters
            height: Height in meters
            resolution: Grid resolution in meters
            config: Terrain configuration
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.config = config or {}
        
        # Grid dimensions
        self.nx = int(width / resolution) + 1
        self.ny = int(height / resolution) + 1
        
        # Terrain data arrays
        self.elevation = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.terrain_type = np.zeros((self.ny, self.nx), dtype=np.int8)
        
        # Cached derivatives
        self.gradient_x = None
        self.gradient_y = None
        self.normals = None
        
        # Statistics
        self.min_elevation = 0.0
        self.max_elevation = 0.0
        self.mean_elevation = 0.0
        
        # Generator
        self.generator = TerrainGenerator(config.get('parameters', {}))
    
    def generate(self, seed: Optional[int] = None):
        """
        Generate terrain using configured method.
        
        Args:
            seed: Random seed for generation
        """
        method = self.config.get('generator', 'perlin')
        
        if seed is None:
            seed = self.config.get('seed', 42)
        
        # Generate elevation data
        if method == 'perlin':
            self.elevation = self.generator.generate_perlin(
                self.nx, self.ny, seed
            )
        elif method == 'diamond_square':
            self.elevation = self.generator.generate_diamond_square(
                self.nx, self.ny, seed
            )
        elif method == 'flat':
            self.elevation = np.full((self.ny, self.nx), 
                                    self.config.get('parameters', {}).get('elevation', 100))
        else:
            raise ValueError(f"Unknown terrain generation method: {method}")
        
        # Classify terrain types
        self._classify_terrain_types()
        
        # Update statistics
        self._update_statistics()
        
        # Clear cached derivatives (will be recalculated on demand)
        self.gradient_x = None
        self.gradient_y = None
        self.normals = None
    
    def _classify_terrain_types(self):
        """Classify terrain types based on elevation."""
        params = self.config.get('parameters', {})
        water_level = params.get('water_level', 0)
        grass_limit = params.get('grass_limit', 500)
        dirt_limit = params.get('dirt_limit', 1000)
        rock_limit = params.get('rock_limit', 1500)
        
        self.terrain_type = np.zeros_like(self.elevation, dtype=np.int8)
        self.terrain_type[self.elevation <= water_level] = TerrainType.WATER
        self.terrain_type[(self.elevation > water_level) & 
                         (self.elevation <= grass_limit)] = TerrainType.GRASS
        self.terrain_type[(self.elevation > grass_limit) & 
                         (self.elevation <= dirt_limit)] = TerrainType.DIRT
        self.terrain_type[(self.elevation > dirt_limit) & 
                         (self.elevation <= rock_limit)] = TerrainType.ROCK
        self.terrain_type[self.elevation > rock_limit] = TerrainType.SNOW
    
    def _update_statistics(self):
        """Update terrain statistics."""
        self.min_elevation = float(np.min(self.elevation))
        self.max_elevation = float(np.max(self.elevation))
        self.mean_elevation = float(np.mean(self.elevation))
    
    def get_elevation_at(self, x: float, y: float) -> float:
        """
        Get interpolated elevation at exact position.
        
        Args:
            x: East coordinate (meters)
            y: North coordinate (meters)
            
        Returns:
            Elevation in meters
        """
        # Convert to grid coordinates
        gx = x / self.resolution
        gy = y / self.resolution
        
        # Clamp to grid bounds
        gx = np.clip(gx, 0, self.nx - 1)
        gy = np.clip(gy, 0, self.ny - 1)
        
        # Get integer indices
        ix = int(gx)
        iy = int(gy)
        
        # Handle edge cases
        if ix == self.nx - 1:
            ix = self.nx - 2
        if iy == self.ny - 1:
            iy = self.ny - 2
        
        # Get fractional parts
        fx = gx - ix
        fy = gy - iy
        
        # Get corner elevations
        z00 = self.elevation[iy, ix]
        z10 = self.elevation[iy, ix + 1]
        z01 = self.elevation[iy + 1, ix]
        z11 = self.elevation[iy + 1, ix + 1]
        
        # Bilinear interpolation
        return bilinear_interpolate(z00, z10, z01, z11, fx, fy)
    
    def get_gradient(self, x: float, y: float) -> Tuple[float, float]:
        """
        Get terrain gradient at position.
        
        Args:
            x: East coordinate (meters)
            y: North coordinate (meters)
            
        Returns:
            Gradient (dz/dx, dz/dy)
        """
        # Calculate gradients if not cached
        if self.gradient_x is None or self.gradient_y is None:
            self._calculate_gradients()
        
        # Convert to grid coordinates
        gx = x / self.resolution
        gy = y / self.resolution
        
        # Clamp and get indices
        ix = int(np.clip(gx, 0, self.nx - 1))
        iy = int(np.clip(gy, 0, self.ny - 1))
        
        return (self.gradient_x[iy, ix], self.gradient_y[iy, ix])
    
    def get_normal(self, x: float, y: float) -> np.ndarray:
        """
        Get surface normal vector at position.
        
        Args:
            x: East coordinate (meters)
            y: North coordinate (meters)
            
        Returns:
            Normal vector (nx, ny, nz)
        """
        # Calculate normals if not cached
        if self.normals is None:
            self._calculate_normals()
        
        # Convert to grid coordinates
        gx = x / self.resolution
        gy = y / self.resolution
        
        # Clamp and get indices
        ix = int(np.clip(gx, 0, self.nx - 1))
        iy = int(np.clip(gy, 0, self.ny - 1))
        
        return self.normals[iy, ix]
    
    def _calculate_gradients(self):
        """Calculate terrain gradients."""
        # Use numpy gradient with proper spacing
        self.gradient_y, self.gradient_x = np.gradient(
            self.elevation, self.resolution, self.resolution
        )
    
    def _calculate_normals(self):
        """Calculate surface normals."""
        if self.gradient_x is None or self.gradient_y is None:
            self._calculate_gradients()
        
        # Initialize normal array
        self.normals = np.zeros((self.ny, self.nx, 3), dtype=np.float32)
        
        # Calculate normals from gradients
        # Normal = (-dz/dx, -dz/dy, 1) normalized
        self.normals[:, :, 0] = -self.gradient_x
        self.normals[:, :, 1] = -self.gradient_y
        self.normals[:, :, 2] = 1.0
        
        # Normalize
        norm = np.linalg.norm(self.normals, axis=2, keepdims=True)
        self.normals /= norm
    
    def get_slope(self, x: float, y: float) -> float:
        """
        Get terrain slope angle at position.
        
        Args:
            x: East coordinate (meters)
            y: North coordinate (meters)
            
        Returns:
            Slope angle in radians
        """
        normal = self.get_normal(x, y)
        # Slope is angle between normal and vertical
        return np.arccos(np.clip(normal[2], -1, 1))
    
    def get_terrain_type_at(self, x: float, y: float) -> TerrainType:
        """
        Get terrain type at position.
        
        Args:
            x: East coordinate (meters)
            y: North coordinate (meters)
            
        Returns:
            Terrain type enum
        """
        # Convert to grid coordinates
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        
        # Clamp to grid bounds
        gx = np.clip(gx, 0, self.nx - 1)
        gy = np.clip(gy, 0, self.ny - 1)
        
        return TerrainType(self.terrain_type[gy, gx])
    
    def get_minimum_safe_altitude(self, x: float, y: float, 
                                 radius: float = 1000.0,
                                 safety_margin: float = 100.0) -> float:
        """
        Get minimum safe altitude considering nearby terrain.
        
        Args:
            x: East coordinate (meters)
            y: North coordinate (meters)
            radius: Search radius (meters)
            safety_margin: Safety margin above terrain (meters)
            
        Returns:
            Minimum safe altitude in meters
        """
        # Convert radius to grid cells
        cells = int(radius / self.resolution)
        
        # Get center grid position
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        
        # Define search bounds
        x_min = max(0, gx - cells)
        x_max = min(self.nx - 1, gx + cells)
        y_min = max(0, gy - cells)
        y_max = min(self.ny - 1, gy + cells)
        
        # Find maximum elevation in area
        area_elevation = self.elevation[y_min:y_max+1, x_min:x_max+1]
        max_elevation = np.max(area_elevation)
        
        return max_elevation + safety_margin
    
    def get_elevation_profile(self, start: Tuple[float, float], 
                             end: Tuple[float, float],
                             num_samples: int = 100) -> np.ndarray:
        """
        Get elevation profile along a line.
        
        Args:
            start: Start position (x, y)
            end: End position (x, y)
            num_samples: Number of sample points
            
        Returns:
            Array of elevations along the line
        """
        x_samples = np.linspace(start[0], end[0], num_samples)
        y_samples = np.linspace(start[1], end[1], num_samples)
        
        elevations = np.zeros(num_samples)
        for i in range(num_samples):
            elevations[i] = self.get_elevation_at(x_samples[i], y_samples[i])
        
        return elevations