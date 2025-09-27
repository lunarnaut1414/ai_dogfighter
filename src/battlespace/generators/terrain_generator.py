"""
Terrain generation algorithms for battlespace.
Implements various procedural generation methods.
"""

import numpy as np
from noise import pnoise2
from typing import Optional, Dict, Any


class TerrainGenerator:
    """
    Handles procedural terrain generation using various algorithms.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize terrain generator.
        
        Args:
            config: Generation parameters
        """
        self.config = config or {}
        
        # Default parameters
        self.octaves = config.get('octaves', 6)
        self.frequency = config.get('frequency', 0.0001)
        self.amplitude = config.get('amplitude', 2000)
        self.base_elevation = config.get('base_elevation', 100)
        self.persistence = config.get('persistence', 0.5)
        self.lacunarity = config.get('lacunarity', 2.0)
    
    def generate_perlin(self, width: int, height: int, 
                       seed: int = 42) -> np.ndarray:
        """
        Generate terrain using Perlin noise.
        
        Args:
            width: Grid width
            height: Grid height
            seed: Random seed
            
        Returns:
            2D elevation array
        """
        terrain = np.zeros((height, width), dtype=np.float32)
        
        # Use seed as offset for perlin noise
        x_offset = seed * 1000
        y_offset = seed * 2000
        
        for y in range(height):
            for x in range(width):
                # Sample Perlin noise at multiple octaves
                elevation = 0.0
                amplitude = self.amplitude
                frequency = self.frequency
                
                for _ in range(self.octaves):
                    nx = (x + x_offset) * frequency
                    ny = (y + y_offset) * frequency
                    
                    noise_value = pnoise2(nx, ny)
                    elevation += noise_value * amplitude
                    
                    amplitude *= self.persistence
                    frequency *= self.lacunarity
                
                terrain[y, x] = self.base_elevation + elevation
        
        # Apply smoothing
        terrain = self._smooth_terrain(terrain)
        
        # Ensure minimum elevation
        terrain = np.maximum(terrain, 0)
        
        return terrain
    
    def generate_diamond_square(self, width: int, height: int,
                               seed: int = 42) -> np.ndarray:
        """
        Generate terrain using diamond-square algorithm.
        
        Args:
            width: Grid width (should be 2^n + 1)
            height: Grid height (should be 2^n + 1)
            seed: Random seed
            
        Returns:
            2D elevation array
        """
        np.random.seed(seed)
        
        # Adjust dimensions to nearest power of 2 + 1
        n = max(int(np.log2(max(width, height) - 1)), 1)
        size = 2**n + 1
        
        terrain = np.zeros((size, size), dtype=np.float32)
        
        # Initialize corners
        terrain[0, 0] = self._random_elevation()
        terrain[0, size-1] = self._random_elevation()
        terrain[size-1, 0] = self._random_elevation()
        terrain[size-1, size-1] = self._random_elevation()
        
        # Diamond-square iterations
        step = size - 1
        roughness = self.amplitude
        
        while step > 1:
            half_step = step // 2
            
            # Diamond step
            for y in range(half_step, size, step):
                for x in range(half_step, size, step):
                    average = (
                        terrain[y - half_step, x - half_step] +
                        terrain[y - half_step, x + half_step] +
                        terrain[y + half_step, x - half_step] +
                        terrain[y + half_step, x + half_step]
                    ) / 4.0
                    
                    terrain[y, x] = average + np.random.randn() * roughness
            
            # Square step
            for y in range(0, size, half_step):
                for x in range((y + half_step) % step, size, step):
                    count = 0
                    total = 0
                    
                    if y - half_step >= 0:
                        total += terrain[y - half_step, x]
                        count += 1
                    if y + half_step < size:
                        total += terrain[y + half_step, x]
                        count += 1
                    if x - half_step >= 0:
                        total += terrain[y, x - half_step]
                        count += 1
                    if x + half_step < size:
                        total += terrain[y, x + half_step]
                        count += 1
                    
                    terrain[y, x] = total / count + np.random.randn() * roughness
            
            step = half_step
            roughness *= self.persistence
        
        # Crop to requested size
        terrain = terrain[:height, :width]
        
        # Add base elevation
        terrain += self.base_elevation
        
        # Ensure minimum elevation
        terrain = np.maximum(terrain, 0)
        
        return terrain
    
    def generate_ridged_noise(self, width: int, height: int,
                             seed: int = 42) -> np.ndarray:
        """
        Generate ridged terrain (mountain ridges).
        
        Args:
            width: Grid width
            height: Grid height
            seed: Random seed
            
        Returns:
            2D elevation array
        """
        terrain = np.zeros((height, width), dtype=np.float32)
        
        x_offset = seed * 1000
        y_offset = seed * 2000
        
        for y in range(height):
            for x in range(width):
                elevation = 0.0
                amplitude = self.amplitude
                frequency = self.frequency
                weight = 1.0
                
                for _ in range(self.octaves):
                    nx = (x + x_offset) * frequency
                    ny = (y + y_offset) * frequency
                    
                    # Ridged noise: 1 - abs(noise)
                    noise_value = 1.0 - abs(pnoise2(nx, ny))
                    noise_value = noise_value * noise_value * weight
                    
                    weight = noise_value * 2.0
                    weight = np.clip(weight, 0, 1)
                    
                    elevation += noise_value * amplitude
                    
                    amplitude *= self.persistence
                    frequency *= self.lacunarity
                
                terrain[y, x] = self.base_elevation + elevation
        
        terrain = np.maximum(terrain, 0)
        return terrain
    
    def generate_islands(self, width: int, height: int,
                        seed: int = 42) -> np.ndarray:
        """
        Generate island-style terrain with water.
        
        Args:
            width: Grid width
            height: Grid height
            seed: Random seed
            
        Returns:
            2D elevation array
        """
        # First generate base terrain
        terrain = self.generate_perlin(width, height, seed)
        
        # Create radial gradient for island shape
        center_x = width / 2
        center_y = height / 2
        max_radius = min(width, height) / 2
        
        for y in range(height):
            for x in range(width):
                # Distance from center
                dx = x - center_x
                dy = y - center_y
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Radial falloff
                falloff = 1.0 - (distance / max_radius)
                falloff = np.clip(falloff, 0, 1)
                falloff = falloff ** 2  # Smooth falloff
                
                # Apply falloff to terrain
                terrain[y, x] *= falloff
                
                # Set water level
                if terrain[y, x] < 10:
                    terrain[y, x] = -50  # Below sea level
        
        return terrain
    
    def _random_elevation(self) -> float:
        """Generate random elevation value."""
        return self.base_elevation + np.random.randn() * self.amplitude * 0.5
    
    def _smooth_terrain(self, terrain: np.ndarray, 
                       iterations: int = 1) -> np.ndarray:
        """
        Apply smoothing filter to terrain.
        
        Args:
            terrain: Input terrain array
            iterations: Number of smoothing passes
            
        Returns:
            Smoothed terrain
        """
        kernel = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]]) / 16.0
        
        from scipy.ndimage import convolve
        
        for _ in range(iterations):
            terrain = convolve(terrain, kernel, mode='reflect')
        
        return terrain
    
    def add_feature(self, terrain: np.ndarray, feature_type: str,
                   position: tuple, params: dict) -> np.ndarray:
        """
        Add specific terrain feature.
        
        Args:
            terrain: Existing terrain array
            feature_type: Type of feature ('mountain', 'valley', 'plateau')
            position: (x, y) position for feature
            params: Feature-specific parameters
            
        Returns:
            Modified terrain
        """
        x, y = position
        
        if feature_type == 'mountain':
            radius = params.get('radius', 500)
            height = params.get('height', 1000)
            
            for j in range(terrain.shape[0]):
                for i in range(terrain.shape[1]):
                    dx = i - x
                    dy = j - y
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < radius:
                        # Gaussian mountain
                        elevation = height * np.exp(-(dist/radius)**2)
                        terrain[j, i] += elevation
        
        elif feature_type == 'valley':
            length = params.get('length', 1000)
            width = params.get('width', 200)
            depth = params.get('depth', 100)
            angle = params.get('angle', 0)
            
            # Create valley along specified angle
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            for j in range(terrain.shape[0]):
                for i in range(terrain.shape[1]):
                    # Rotate coordinates
                    dx = i - x
                    dy = j - y
                    rx = dx * cos_a + dy * sin_a
                    ry = -dx * sin_a + dy * cos_a
                    
                    if abs(rx) < length/2 and abs(ry) < width/2:
                        # Valley profile
                        valley_depth = depth * (1 - (ry/width)**2)
                        terrain[j, i] -= valley_depth
        
        elif feature_type == 'plateau':
            radius = params.get('radius', 500)
            height = params.get('height', 500)
            
            for j in range(terrain.shape[0]):
                for i in range(terrain.shape[1]):
                    dx = i - x
                    dy = j - y
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < radius:
                        # Flat-topped plateau with smooth edges
                        if dist < radius * 0.8:
                            terrain[j, i] += height
                        else:
                            # Smooth transition
                            t = (dist - radius * 0.8) / (radius * 0.2)
                            terrain[j, i] += height * (1 - t)
        
        return terrain