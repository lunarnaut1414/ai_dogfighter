"""
Core Battlespace environment management with integrated features.
Main container for all environment layers and systems.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import yaml
from pathlib import Path

from .terrain import TerrainLayer, TerrainType
from .structures import StructureLayer
from .airspace import AirspaceLayer
from .weather import WeatherSystem
from .utils.coordinate_utils import validate_position, world_to_grid


class Battlespace:
    """
    Main battlespace environment container with integrated terrain-weather effects.
    Manages terrain, structures, airspace, and weather systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 config_file: Optional[str] = None,
                 enable_integration: bool = True):
        """
        Initialize battlespace from configuration.
        
        Args:
            config: Configuration dictionary
            config_file: Path to YAML configuration file
            enable_integration: Enable terrain-weather integration
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
        
        # Integration flag
        self.integration_enabled = enable_integration
        
        # Initialize layers
        self._initialize_layers()
        
        # Cache for performance
        self._elevation_cache = {}
        self._los_cache = {}
        
        # Tactical analysis caches (for integrated features)
        self._radar_shadow_map = None
        self._optimal_paths_cache = {}
        self._thermal_map = None
        
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
        
        # Weather system (will be upgraded to integrated version if needed)
        weather_config = self.config.get('weather', {})
        self.weather = WeatherSystem(
            self.width, self.height, self.altitude_ceiling,
            weather_config
        )
    
    def generate(self, seed: Optional[int] = None):
        """
        Generate all battlespace layers with optional integration.
        
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
        
        # Generate weather (with terrain integration if enabled)
        if self.integration_enabled:
            # Check if weather system has integration capabilities
            if hasattr(self.weather, 'set_terrain_layer'):
                self.weather.set_terrain_layer(self.terrain)
                if hasattr(self.weather, 'generate_with_terrain'):
                    self.weather.generate_with_terrain()
                else:
                    self.weather.generate()
            else:
                self.weather.generate()
        else:
            self.weather.generate()
        
        # Pre-compute tactical features if integration enabled
        if self.integration_enabled:
            self._compute_tactical_features()
        
        # Clear caches after generation
        self._elevation_cache.clear()
        self._los_cache.clear()
        self._optimal_paths_cache.clear()
    
    def _compute_tactical_features(self):
        """Pre-compute tactical terrain features."""
        # Compute radar shadow map (terrain masking)
        self._compute_radar_shadows()
        
        # Identify thermal hotspots
        self._identify_thermal_zones()
        
        # Identify tactical positions
        self._identify_tactical_positions()
    
    def _compute_radar_shadows(self, radar_altitude: float = 5000):
        """
        Compute areas hidden from radar by terrain.
        
        Args:
            radar_altitude: Assumed radar altitude
        """
        nx = self.terrain.nx
        ny = self.terrain.ny
        self._radar_shadow_map = np.zeros((ny, nx), dtype=bool)
        
        # Simplified: Check visibility from center at radar altitude
        radar_pos = np.array([self.width/2, self.height/2, radar_altitude])
        
        for j in range(0, ny, 5):  # Sample every 5 cells for efficiency
            for i in range(0, nx, 5):
                x = i * self.terrain.resolution
                y = j * self.terrain.resolution
                z = self.terrain.elevation[j, i] + 50  # 50m above terrain
                
                target_pos = np.array([x, y, z])
                
                # Check LOS to radar
                if not self.get_line_of_sight(radar_pos, target_pos, sample_distance=200):
                    self._radar_shadow_map[j:j+5, i:i+5] = True
    
    def _identify_thermal_zones(self):
        """Identify areas with strong thermal activity."""
        self._thermal_map = np.zeros((self.terrain.ny, self.terrain.nx))
        
        for j in range(self.terrain.ny):
            for i in range(self.terrain.nx):
                terrain_type = self.terrain.terrain_type[j, i]
                
                # Assign thermal potential
                if terrain_type == TerrainType.ROCK:
                    self._thermal_map[j, i] = 1.5
                elif terrain_type == TerrainType.DIRT:
                    self._thermal_map[j, i] = 1.2
                elif terrain_type == TerrainType.GRASS:
                    self._thermal_map[j, i] = 1.0
                elif terrain_type == TerrainType.WATER:
                    self._thermal_map[j, i] = 0.3
                elif terrain_type == TerrainType.SNOW:
                    self._thermal_map[j, i] = 0.1
    
    def _identify_tactical_positions(self):
        """Identify tactically advantageous positions."""
        self.tactical_positions = {
            'high_ground': [],
            'valleys': [],
            'ridges': [],
            'choke_points': []
        }
        
        # Find high ground positions
        elevation_threshold = np.percentile(self.terrain.elevation, 80)
        high_ground = np.where(self.terrain.elevation > elevation_threshold)
        
        for j, i in zip(high_ground[0], high_ground[1]):
            x = i * self.terrain.resolution
            y = j * self.terrain.resolution
            z = self.terrain.elevation[j, i]
            self.tactical_positions['high_ground'].append(np.array([x, y, z]))
    
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
    
    def get_aircraft_environment_effects(self, position: np.ndarray, 
                                        velocity: np.ndarray) -> Dict[str, Any]:
        """
        Get all environmental effects on aircraft at position.
        
        Args:
            position: Aircraft [x, y, z] position
            velocity: Aircraft [vx, vy, vz] velocity
            
        Returns:
            Dictionary with all environmental effects
        """
        effects = {}
        
        # Terrain effects
        terrain_elevation = self.get_elevation(position[0], position[1])
        agl_altitude = position[2] - terrain_elevation
        effects['agl_altitude'] = agl_altitude
        effects['terrain_elevation'] = terrain_elevation
        
        # Ground effect (increased lift near ground)
        if agl_altitude < 50:  # Within one wingspan typically
            effects['ground_effect_factor'] = 1.0 + (0.2 * (1.0 - agl_altitude/50))
        else:
            effects['ground_effect_factor'] = 1.0
        
        # Wind effects
        wind = self.get_wind(position)
        effects['wind_vector'] = wind
        
        # Calculate relative wind
        relative_wind = velocity - wind
        effects['relative_wind'] = relative_wind
        effects['relative_airspeed'] = np.linalg.norm(relative_wind)
        
        # Turbulence
        turbulence = self.weather.get_turbulence_at(position)
        effects['turbulence_intensity'] = turbulence
        
        # Add random turbulence perturbation
        if turbulence > 0:
            turb_magnitude = turbulence * 5.0  # Scale to m/s
            effects['turbulence_perturbation'] = np.random.randn(3) * turb_magnitude
        else:
            effects['turbulence_perturbation'] = np.zeros(3)
        
        # Air density (affects lift and drag)
        air_density = self.weather.get_density_at(position[2])
        effects['air_density'] = air_density
        effects['density_ratio'] = air_density / 1.225  # Ratio to sea level
        
        # Terrain influence metrics (if weather has integration)
        if hasattr(self.weather, 'get_terrain_influenced_metrics'):
            terrain_metrics = self.weather.get_terrain_influenced_metrics(position)
            effects.update(terrain_metrics)
        
        # Radar visibility (if computed)
        if self._radar_shadow_map is not None:
            grid_x = int(position[0] / self.terrain.resolution)
            grid_y = int(position[1] / self.terrain.resolution)
            if 0 <= grid_x < self.terrain.nx and 0 <= grid_y < self.terrain.ny:
                effects['radar_visible'] = not self._radar_shadow_map[grid_y, grid_x]
            else:
                effects['radar_visible'] = True
        
        return effects
    
    def find_optimal_intercept_path(self, start_pos: np.ndarray, 
                                   target_pos: np.ndarray,
                                   constraints: Optional[Dict] = None) -> List[np.ndarray]:
        """
        Find optimal path considering terrain and wind.
        
        Args:
            start_pos: Starting position
            target_pos: Target position
            constraints: Path constraints (min_altitude, max_climb_rate, etc.)
            
        Returns:
            List of waypoints for optimal path
        """
        # Check cache
        cache_key = (tuple(start_pos), tuple(target_pos))
        if cache_key in self._optimal_paths_cache:
            return self._optimal_paths_cache[cache_key]
        
        constraints = constraints or {}
        min_altitude_agl = constraints.get('min_altitude_agl', 100)
        
        # Simple implementation: straight line with terrain clearance
        num_waypoints = 20
        waypoints = []
        
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)
            
            # Interpolate position
            pos = start_pos + t * (target_pos - start_pos)
            
            # Ensure terrain clearance
            terrain_height = self.get_elevation(pos[0], pos[1])
            min_safe_alt = terrain_height + min_altitude_agl
            
            if pos[2] < min_safe_alt:
                pos[2] = min_safe_alt
            
            waypoints.append(pos.copy())
        
        # Cache result
        self._optimal_paths_cache[cache_key] = waypoints
        
        return waypoints
    
    def find_terrain_masking_route(self, start_pos: np.ndarray,
                                  target_pos: np.ndarray,
                                  threat_pos: np.ndarray) -> List[np.ndarray]:
        """
        Find route that uses terrain to mask from threat.
        
        Args:
            start_pos: Starting position
            target_pos: Target position  
            threat_pos: Threat/radar position to avoid
            
        Returns:
            List of waypoints using terrain masking
        """
        waypoints = []
        
        # Sample potential waypoints in valleys and behind ridges
        num_samples = 50
        best_path = None
        best_exposure = float('inf')
        
        for _ in range(10):  # Try 10 random paths
            path = [start_pos]
            current = start_pos.copy()
            
            # Add random waypoints preferring low terrain
            for _ in range(3):
                # Random point between current and target
                direction = target_pos - current
                distance = np.linalg.norm(direction)
                direction = direction / distance
                
                # Random distance along direction
                step_dist = np.random.uniform(distance * 0.2, distance * 0.5)
                next_point = current + direction * step_dist
                
                # Add lateral deviation toward lower terrain
                lateral = np.array([-direction[1], direction[0], 0])
                next_point += lateral * np.random.randn() * 1000
                
                # Set altitude for terrain following
                terrain_height = self.get_elevation(next_point[0], next_point[1])
                next_point[2] = terrain_height + 100  # 100m AGL
                
                path.append(next_point)
                current = next_point
            
            path.append(target_pos)
            
            # Evaluate exposure to threat
            exposure = 0
            for point in path:
                if self.get_line_of_sight(threat_pos, point):
                    exposure += 1
            
            if exposure < best_exposure:
                best_exposure = exposure
                best_path = path
        
        return best_path if best_path else [start_pos, target_pos]
    
    def get_tactical_advantages(self, position: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate tactical advantages at a position.
        
        Args:
            position: [x, y, z] position to evaluate
            
        Returns:
            Dictionary of tactical factors
        """
        advantages = {}
        
        # Height advantage
        terrain_height = self.get_elevation(position[0], position[1])
        mean_terrain = np.mean(self.terrain.elevation)
        advantages['height_advantage'] = (terrain_height - mean_terrain) / mean_terrain
        
        # Energy advantage (altitude = potential energy)
        advantages['energy_state'] = position[2] / self.altitude_ceiling
        
        # Escape routes (check directions without terrain collision)
        escape_routes = 0
        for angle in np.linspace(0, 2*np.pi, 8):
            escape_pos = position + np.array([np.cos(angle)*1000, np.sin(angle)*1000, 0])
            if not self.check_collision(escape_pos, radius=50):
                escape_routes += 1
        advantages['escape_routes'] = escape_routes
        
        # Terrain masking availability
        if self._radar_shadow_map is not None:
            # Check how much terrain masking is nearby
            grid_x = int(position[0] / self.terrain.resolution)
            grid_y = int(position[1] / self.terrain.resolution)
            
            masking_score = 0
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    gx, gy = grid_x + dx, grid_y + dy
                    if 0 <= gx < self.terrain.nx and 0 <= gy < self.terrain.ny:
                        if self._radar_shadow_map[gy, gx]:
                            masking_score += 1
            advantages['terrain_masking_nearby'] = masking_score / 441  # Normalized
        
        # Wind advantage (tailwind vs headwind for pursuit)
        wind = self.get_wind(position)
        advantages['wind_vector'] = wind
        
        return advantages
    
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
            'num_no_fly_zones': len(self.airspace.no_fly_zones),
            'integration_enabled': self.integration_enabled
        }
    
    def save(self, filepath: str):
        """Save battlespace state to file."""
        # Implementation for saving generated battlespace
        pass
    
    def load(self, filepath: str):
        """Load battlespace state from file."""
        # Implementation for loading saved battlespace
        pass