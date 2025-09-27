"""
Weather system for battlespace environment.
Manages wind fields, turbulence, and atmospheric conditions.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.interpolate import RegularGridInterpolator

from .terrain import TerrainLayer, TerrainType

class WeatherSystem:
    """
    Manages environmental conditions affecting flight.
    """
    
    def __init__(self, width: float, height: float, ceiling: float,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize weather system.
        
        Args:
            width: Battlespace width in meters
            height: Battlespace height in meters
            ceiling: Maximum altitude in meters
            config: Weather configuration
        """
        self.width = width
        self.height = height
        self.ceiling = ceiling
        self.config = config or {}
        
        # Grid resolution for wind field
        self.horizontal_resolution = config.get('horizontal_resolution', 1000)  # 1km
        self.vertical_resolution = config.get('vertical_resolution', 500)  # 500m
        
        # Grid dimensions
        self.nx = int(width / self.horizontal_resolution) + 1
        self.ny = int(height / self.horizontal_resolution) + 1
        self.nz = int(ceiling / self.vertical_resolution) + 1
        
        # Wind field components
        self.wind_u = None  # East component
        self.wind_v = None  # North component
        self.wind_w = None  # Vertical component
        
        # Turbulence intensity field
        self.turbulence = None
        
        # Atmospheric properties
        self.temperature_profile = None
        self.pressure_profile = None
        self.density_profile = None
        
        # Interpolators (created after generation)
        self.wind_interpolator_u = None
        self.wind_interpolator_v = None
        self.wind_interpolator_w = None
        self.turbulence_interpolator = None
        
        # Time-varying parameters
        self.time = 0.0  # Simulation time in seconds
        self.time_of_day = 12.0  # 24-hour format
        self.weather_type = 'clear'  # clear, cloudy, stormy
        
        # Store base configuration for temporal updates
        self.base_wind_config = config.get('wind', {}).copy()
        
    def generate(self, seed: Optional[int] = None):
        """
        Generate weather fields.
        
        Args:
            seed: Random seed for reproducible weather
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Get base wind from config
        wind_config = self.config.get('wind', {})
        base_wind = np.array(wind_config.get('base_vector', [10, 0, 0]))
        altitude_multiplier = wind_config.get('altitude_multiplier', 1.5)
        
        # Initialize wind fields
        self.wind_u = np.zeros((self.nz, self.ny, self.nx))
        self.wind_v = np.zeros((self.nz, self.ny, self.nx))
        self.wind_w = np.zeros((self.nz, self.ny, self.nx))
        
        # Generate wind field with altitude variation
        for k in range(self.nz):
            altitude = k * self.vertical_resolution
            
            # Wind increases with altitude (exponential profile)
            altitude_factor = 1.0 + (altitude_multiplier - 1.0) * (altitude / self.ceiling)
            
            # Base wind at this altitude
            self.wind_u[k, :, :] = base_wind[0] * altitude_factor
            self.wind_v[k, :, :] = base_wind[1] * altitude_factor
            
            # Add spatial variation (simplified turbulence)
            noise_scale = 2.0  # m/s variation
            self.wind_u[k] += np.random.randn(self.ny, self.nx) * noise_scale
            self.wind_v[k] += np.random.randn(self.ny, self.nx) * noise_scale
            
            # Small vertical component (thermals/downdrafts)
            self.wind_w[k] = np.random.randn(self.ny, self.nx) * 0.5
        
        # Generate turbulence field
        self._generate_turbulence()
        
        # Generate atmospheric profiles
        self._generate_atmosphere()
        
        # Create interpolators
        self._create_interpolators()
    
    def _generate_turbulence(self):
        """Generate turbulence intensity field."""
        self.turbulence = np.zeros((self.nz, self.ny, self.nx))
        
        # Base turbulence level
        base_turbulence = self.config.get('turbulence', {}).get('base_level', 0.1)
        
        for k in range(self.nz):
            altitude = k * self.vertical_resolution
            
            # Turbulence decreases with altitude (boundary layer effect)
            if altitude < 1000:  # Strong turbulence in boundary layer
                turb_factor = 1.0
            elif altitude < 3000:  # Moderate turbulence
                turb_factor = 0.5
            else:  # Light turbulence
                turb_factor = 0.2
            
            # Random turbulence patches
            self.turbulence[k] = base_turbulence * turb_factor * (
                1.0 + np.random.randn(self.ny, self.nx) * 0.5
            )
            
            # Ensure non-negative
            self.turbulence[k] = np.maximum(0, self.turbulence[k])
    
    def _generate_atmosphere(self):
        """Generate atmospheric property profiles."""
        # Standard atmosphere model (simplified)
        altitudes = np.linspace(0, self.ceiling, self.nz)
        
        # Temperature profile (ISA model)
        sea_level_temp = 288.15  # Kelvin
        lapse_rate = -0.0065  # K/m
        self.temperature_profile = sea_level_temp + lapse_rate * altitudes
        
        # Pressure profile (exponential decay)
        sea_level_pressure = 101325  # Pa
        scale_height = 8000  # m
        self.pressure_profile = sea_level_pressure * np.exp(-altitudes / scale_height)
        
        # Density profile (from ideal gas law)
        gas_constant = 287.05  # J/(kg·K)
        self.density_profile = self.pressure_profile / (gas_constant * self.temperature_profile)
    
    def _create_interpolators(self):
        """Create interpolators for efficient queries."""
        # Create coordinate arrays
        x = np.linspace(0, self.width, self.nx)
        y = np.linspace(0, self.height, self.ny)
        z = np.linspace(0, self.ceiling, self.nz)
        
        # Create interpolators for wind components
        self.wind_interpolator_u = RegularGridInterpolator(
            (z, y, x), self.wind_u, bounds_error=False, fill_value=0
        )
        self.wind_interpolator_v = RegularGridInterpolator(
            (z, y, x), self.wind_v, bounds_error=False, fill_value=0
        )
        self.wind_interpolator_w = RegularGridInterpolator(
            (z, y, x), self.wind_w, bounds_error=False, fill_value=0
        )
        
        # Create turbulence interpolator
        self.turbulence_interpolator = RegularGridInterpolator(
            (z, y, x), self.turbulence, bounds_error=False, fill_value=0
        )
    
    def get_wind_at(self, position: np.ndarray) -> np.ndarray:
        """
        Get interpolated wind vector at position.
        
        Args:
            position: [x, y, z] position in meters
            
        Returns:
            Wind vector [vx, vy, vz] in m/s
        """
        if self.wind_interpolator_u is None:
            # No wind field generated yet
            return np.zeros(3)
        
        # Ensure position is within bounds
        pos = np.array([
            np.clip(position[2], 0, self.ceiling),
            np.clip(position[1], 0, self.height),
            np.clip(position[0], 0, self.width)
        ])
        
        # Interpolate wind components
        wind = np.array([
            float(self.wind_interpolator_u(pos)),
            float(self.wind_interpolator_v(pos)),
            float(self.wind_interpolator_w(pos))
        ])
        
        return wind
    
    def get_turbulence_at(self, position: np.ndarray) -> float:
        """
        Get turbulence intensity at position.
        
        Args:
            position: [x, y, z] position in meters
            
        Returns:
            Turbulence intensity (0.0 = calm, 1.0 = severe)
        """
        if self.turbulence_interpolator is None:
            return 0.0
        
        # Ensure position is within bounds
        pos = np.array([
            np.clip(position[2], 0, self.ceiling),
            np.clip(position[1], 0, self.height),
            np.clip(position[0], 0, self.width)
        ])
        
        return float(self.turbulence_interpolator(pos))
    
    def get_density_at(self, altitude: float) -> float:
        """
        Get air density at altitude.
        
        Args:
            altitude: Altitude in meters
            
        Returns:
            Air density in kg/m³
        """
        if self.density_profile is None:
            return 1.225  # Sea level standard
        
        # Linear interpolation in density profile
        idx = int(altitude / self.vertical_resolution)
        if idx >= len(self.density_profile) - 1:
            return self.density_profile[-1]
        if idx < 0:
            return self.density_profile[0]
        
        # Interpolate between levels
        frac = (altitude / self.vertical_resolution) - idx
        return (1 - frac) * self.density_profile[idx] + frac * self.density_profile[idx + 1]
    
    def get_temperature_at(self, altitude: float) -> float:
        """
        Get temperature at altitude.
        
        Args:
            altitude: Altitude in meters
            
        Returns:
            Temperature in Kelvin
        """
        if self.temperature_profile is None:
            return 288.15  # Sea level standard
        
        # Linear interpolation
        idx = int(altitude / self.vertical_resolution)
        if idx >= len(self.temperature_profile) - 1:
            return self.temperature_profile[-1]
        if idx < 0:
            return self.temperature_profile[0]
        
        frac = (altitude / self.vertical_resolution) - idx
        return (1 - frac) * self.temperature_profile[idx] + frac * self.temperature_profile[idx + 1]
    
    def get_pressure_at(self, altitude: float) -> float:
        """
        Get atmospheric pressure at altitude.
        
        Args:
            altitude: Altitude in meters
            
        Returns:
            Pressure in Pascals
        """
        if self.pressure_profile is None:
            return 101325  # Sea level standard
        
        # Linear interpolation
        idx = int(altitude / self.vertical_resolution)
        if idx >= len(self.pressure_profile) - 1:
            return self.pressure_profile[-1]
        if idx < 0:
            return self.pressure_profile[0]
        
        frac = (altitude / self.vertical_resolution) - idx
        return (1 - frac) * self.pressure_profile[idx] + frac * self.pressure_profile[idx + 1]
    
    def add_thermal(self, center: np.ndarray, radius: float, 
                    strength: float, max_altitude: float):
        """
        Add a thermal (updraft) to the wind field.
        
        Args:
            center: [x, y] center position
            radius: Thermal radius in meters
            strength: Maximum vertical velocity in m/s
            max_altitude: Maximum altitude of thermal effect
        """
        # Update wind field with thermal
        for k in range(self.nz):
            altitude = k * self.vertical_resolution
            if altitude > max_altitude:
                break
            
            # Altitude factor (thermal weakens with height)
            alt_factor = 1.0 - (altitude / max_altitude)
            
            for j in range(self.ny):
                for i in range(self.nx):
                    x = i * self.horizontal_resolution
                    y = j * self.horizontal_resolution
                    
                    # Distance from thermal center
                    dx = x - center[0]
                    dy = y - center[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < radius:
                        # Gaussian profile
                        thermal_strength = strength * alt_factor * np.exp(-(dist/radius)**2)
                        self.wind_w[k, j, i] += thermal_strength
        
        # Recreate interpolators
        self._create_interpolators()
    
    def add_wind_shear(self, altitude: float, shear_strength: float):
        """
        Add wind shear layer at specified altitude.
        
        Args:
            altitude: Altitude of shear layer
            shear_strength: Change in wind speed across layer
        """
        # Find altitude index
        k = int(altitude / self.vertical_resolution)
        if k < 1 or k >= self.nz - 1:
            return
        
        # Add shear (sudden change in horizontal wind)
        self.wind_u[k] += shear_strength
        self.wind_u[k+1] -= shear_strength
        
        # Increase turbulence at shear layer
        self.turbulence[k] *= 2.0
        self.turbulence[k+1] *= 2.0
        
        # Recreate interpolators
        self._create_interpolators()
    
    def update(self, dt: float):
        """
        Update weather system with temporal variations.
        
        Args:
            dt: Time step in seconds
        """
        self.time += dt
        
        # Get base wind configuration
        base_wind = np.array(self.base_wind_config.get('base_vector', [10, 0, 0]))
        altitude_multiplier = self.base_wind_config.get('altitude_multiplier', 1.5)
        
        # Time-based variations
        # 1. Slow wind direction rotation (full rotation every 2 hours)
        rotation_period = 7200  # seconds
        angle = 2 * np.pi * (self.time / rotation_period)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Rotate base wind vector
        rotated_wind = np.array([
            base_wind[0] * cos_a - base_wind[1] * sin_a,
            base_wind[0] * sin_a + base_wind[1] * cos_a,
            base_wind[2]
        ])
        
        # 2. Daily thermal cycle (stronger winds during day)
        hour_of_day = (self.time_of_day + self.time/3600) % 24
        # Peak at 2 PM (14:00), minimum at 2 AM (02:00)
        thermal_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)
        
        # 3. Gusting cycles (5-minute period)
        gust_period = 300  # seconds
        gust_factor = 1.0 + 0.2 * np.sin(2 * np.pi * self.time / gust_period)
        
        # Update wind field
        for k in range(self.nz):
            altitude = k * self.vertical_resolution
            
            # Wind increases with altitude
            altitude_factor = 1.0 + (altitude_multiplier - 1.0) * (altitude / self.ceiling)
            
            # Combine all factors
            wind_strength = altitude_factor * thermal_factor * gust_factor
            
            # Apply to wind field
            self.wind_u[k, :, :] = rotated_wind[0] * wind_strength
            self.wind_v[k, :, :] = rotated_wind[1] * wind_strength
            
            # Add spatial variation (turbulence patches that evolve)
            # Use time-based seed for evolving noise
            np.random.seed(int(self.time) + k)
            noise_scale = 2.0 * (1.0 + 0.5 * np.sin(2 * np.pi * self.time / 120))  # 2-min cycle
            self.wind_u[k] += np.random.randn(self.ny, self.nx) * noise_scale
            self.wind_v[k] += np.random.randn(self.ny, self.nx) * noise_scale
            
            # Vertical component - thermals during day
            if 10 < hour_of_day < 18:  # Daytime hours
                # Stronger updrafts during day
                thermal_strength = 2.0 * np.sin(np.pi * (hour_of_day - 10) / 8)
                self.wind_w[k] = np.random.randn(self.ny, self.nx) * thermal_strength
                
                # Add some coherent thermal columns
                if altitude < 3000:  # Thermals below 3km
                    for _ in range(3):  # Add 3 thermal spots
                        thermal_x = int(np.random.uniform(0, self.nx))
                        thermal_y = int(np.random.uniform(0, self.ny))
                        if 0 < thermal_x < self.nx-1 and 0 < thermal_y < self.ny-1:
                            self.wind_w[k, thermal_y-1:thermal_y+2, thermal_x-1:thermal_x+2] += thermal_strength * 3
            else:
                # Weak downdrafts at night
                self.wind_w[k] = np.random.randn(self.ny, self.nx) * -0.3
        
        # Update turbulence based on wind shear and time of day
        self._update_temporal_turbulence(hour_of_day)
        
        # Recreate interpolators with new data
        self._create_interpolators()
    
    def _update_temporal_turbulence(self, hour_of_day: float):
        """Update turbulence field based on time of day."""
        base_turbulence = self.config.get('turbulence', {}).get('base_level', 0.1)
        
        # Stronger turbulence during afternoon
        if 12 < hour_of_day < 18:
            turb_multiplier = 1.5
        elif 6 < hour_of_day < 12:
            turb_multiplier = 1.2
        else:
            turb_multiplier = 0.8
        
        for k in range(self.nz):
            altitude = k * self.vertical_resolution
            
            # Boundary layer turbulence
            if altitude < 1000:
                turb_factor = 1.0 * turb_multiplier
            elif altitude < 3000:
                turb_factor = 0.5 * turb_multiplier
            else:
                turb_factor = 0.2 * turb_multiplier
            
            # Update turbulence field
            self.turbulence[k] = base_turbulence * turb_factor * (
                1.0 + np.random.randn(self.ny, self.nx) * 0.5
            )
            self.turbulence[k] = np.maximum(0, self.turbulence[k])
    
    def get_wind_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of current wind field.
        
        Returns:
            Dictionary with wind statistics
        """
        # Sample wind at different altitudes
        altitudes = [0, 1000, 5000, 10000]
        wind_samples = {}
        
        for alt in altitudes:
            # Center of map
            pos = np.array([self.width/2, self.height/2, alt])
            wind = self.get_wind_at(pos)
            wind_samples[f'{alt}m'] = {
                'speed': float(np.linalg.norm(wind[:2])),  # Horizontal wind speed
                'direction': float(np.degrees(np.arctan2(wind[1], wind[0]))),
                'vertical': float(wind[2])
            }
        
        return {
            'time': self.time,
            'hour_of_day': (self.time_of_day + self.time/3600) % 24,
            'wind_samples': wind_samples,
            'mean_surface_wind': float(np.mean(np.sqrt(self.wind_u[0]**2 + self.wind_v[0]**2))),
            'max_surface_wind': float(np.max(np.sqrt(self.wind_u[0]**2 + self.wind_v[0]**2)))
        }
        
class IntegratedWeatherSystem(WeatherSystem):
    """
    Weather system with terrain-influenced wind patterns.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terrain_layer = None
        self.terrain_influence_enabled = True
        
        # Terrain effect parameters
        self.mountain_wave_strength = 5.0  # m/s updraft strength
        self.valley_channeling_factor = 1.5  # Wind acceleration in valleys
        self.ridge_acceleration_factor = 1.3  # Wind acceleration over ridges
        self.lee_turbulence_factor = 2.0  # Turbulence multiplier on lee side
        
    def set_terrain_layer(self, terrain_layer: TerrainLayer):
        """
        Link terrain layer for terrain-influenced wind.
        
        Args:
            terrain_layer: TerrainLayer object
        """
        self.terrain_layer = terrain_layer
        
        # Pre-compute terrain influence maps
        self._compute_terrain_influence_maps()
    
    def _compute_terrain_influence_maps(self):
        """Pre-compute terrain influence for efficiency."""
        if self.terrain_layer is None:
            return
        
        # Sample terrain at weather grid resolution
        self.terrain_elevation_grid = np.zeros((self.ny, self.nx))
        self.terrain_gradient_x = np.zeros((self.ny, self.nx))
        self.terrain_gradient_y = np.zeros((self.ny, self.nx))
        self.terrain_type_grid = np.zeros((self.ny, self.nx), dtype=int)
        
        for j in range(self.ny):
            for i in range(self.nx):
                x = i * self.horizontal_resolution
                y = j * self.horizontal_resolution
                
                # Get terrain properties
                self.terrain_elevation_grid[j, i] = self.terrain_layer.get_elevation_at(x, y)
                grad_x, grad_y = self.terrain_layer.get_gradient(x, y)
                self.terrain_gradient_x[j, i] = grad_x
                self.terrain_gradient_y[j, i] = grad_y
                self.terrain_type_grid[j, i] = self.terrain_layer.get_terrain_type_at(x, y)
        
        # Smooth gradients for stability
        self.terrain_gradient_x = gaussian_filter(self.terrain_gradient_x, sigma=1)
        self.terrain_gradient_y = gaussian_filter(self.terrain_gradient_y, sigma=1)
        
        # Identify terrain features
        self._identify_terrain_features()
    
    def _identify_terrain_features(self):
        """Identify ridges, valleys, and other features."""
        # Ridge detection (high points with steep gradients)
        gradient_magnitude = np.sqrt(self.terrain_gradient_x**2 + self.terrain_gradient_y**2)
        
        # Ridges: high elevation + moderate gradient
        elevation_normalized = (self.terrain_elevation_grid - self.terrain_elevation_grid.min()) / \
                              (self.terrain_elevation_grid.max() - self.terrain_elevation_grid.min() + 1e-6)
        self.ridge_mask = (elevation_normalized > 0.7) & (gradient_magnitude > 0.05)
        
        # Valleys: low elevation + surrounded by higher terrain
        self.valley_mask = elevation_normalized < 0.3
        
        # Steep slopes for mountain wave generation
        self.steep_slope_mask = gradient_magnitude > 0.1
    
    def generate_with_terrain(self, seed: Optional[int] = None):
        """
        Generate weather fields with terrain influence.
        
        Args:
            seed: Random seed for reproducible weather
        """
        # First generate base weather
        self.generate(seed)
        
        if self.terrain_layer is None or not self.terrain_influence_enabled:
            return
        
        # Apply terrain modifications
        self._apply_mountain_waves()
        self._apply_valley_channeling()
        self._apply_ridge_acceleration()
        self._apply_thermal_terrain_variation()
        self._apply_lee_turbulence()
        
        # Recreate interpolators with modified fields
        self._create_interpolators()
    
    def _apply_mountain_waves(self):
        """Generate mountain waves from terrain."""
        for j in range(self.ny):
            for i in range(self.nx):
                if self.steep_slope_mask[j, i]:
                    # Check if wind is hitting the slope
                    wind_dir = np.array([self.wind_u[0, j, i], self.wind_v[0, j, i]])
                    gradient = np.array([self.terrain_gradient_x[j, i], 
                                       self.terrain_gradient_y[j, i]])
                    
                    # Dot product: negative means wind hitting upslope
                    wind_slope_alignment = np.dot(wind_dir, gradient)
                    
                    if wind_slope_alignment < 0:  # Windward side
                        # Generate updrafts
                        updraft_strength = abs(wind_slope_alignment) * self.mountain_wave_strength
                        
                        # Apply updraft at multiple altitudes
                        terrain_height = self.terrain_elevation_grid[j, i]
                        for k in range(self.nz):
                            altitude = k * self.vertical_resolution
                            
                            # Updraft strongest near terrain, decreases with altitude
                            if altitude < terrain_height + 2000:  # Effect up to 2km above terrain
                                height_factor = 1.0 - (altitude - terrain_height) / 2000
                                height_factor = max(0, height_factor)
                                
                                self.wind_w[k, j, i] += updraft_strength * height_factor
                                
                                # Add some horizontal deflection
                                deflection = gradient / (np.linalg.norm(gradient) + 1e-6)
                                self.wind_u[k, j, i] -= deflection[0] * updraft_strength * 0.3
                                self.wind_v[k, j, i] -= deflection[1] * updraft_strength * 0.3
                    
                    else:  # Leeward side
                        # Generate downdrafts and turbulence
                        downdraft_strength = abs(wind_slope_alignment) * self.mountain_wave_strength * 0.5
                        
                        for k in range(min(5, self.nz)):  # Lower altitudes only
                            self.wind_w[k, j, i] -= downdraft_strength * 0.5
    
    def _apply_valley_channeling(self):
        """Accelerate wind through valleys."""
        for j in range(1, self.ny-1):
            for i in range(1, self.nx-1):
                if self.valley_mask[j, i]:
                    # Check surrounding terrain
                    surrounding_heights = [
                        self.terrain_elevation_grid[j-1, i],
                        self.terrain_elevation_grid[j+1, i],
                        self.terrain_elevation_grid[j, i-1],
                        self.terrain_elevation_grid[j, i+1]
                    ]
                    center_height = self.terrain_elevation_grid[j, i]
                    
                    # If surrounded by higher terrain, channel the wind
                    if all(h > center_height + 50 for h in surrounding_heights):
                        # Determine valley orientation
                        valley_dir_x = (self.terrain_elevation_grid[j, i+1] - 
                                       self.terrain_elevation_grid[j, i-1])
                        valley_dir_y = (self.terrain_elevation_grid[j+1, i] - 
                                       self.terrain_elevation_grid[j-1, i])
                        
                        # Normalize
                        valley_dir = np.array([valley_dir_x, valley_dir_y])
                        valley_length = np.linalg.norm(valley_dir)
                        if valley_length > 0:
                            valley_dir /= valley_length
                            
                            # Align wind with valley
                            for k in range(min(3, self.nz)):  # Lower altitudes
                                current_wind = np.array([self.wind_u[k, j, i], 
                                                        self.wind_v[k, j, i]])
                                wind_speed = np.linalg.norm(current_wind)
                                
                                # Project wind onto valley direction and amplify
                                aligned_wind = valley_dir * wind_speed * self.valley_channeling_factor
                                
                                # Blend with original wind
                                self.wind_u[k, j, i] = 0.7 * aligned_wind[0] + 0.3 * self.wind_u[k, j, i]
                                self.wind_v[k, j, i] = 0.7 * aligned_wind[1] + 0.3 * self.wind_v[k, j, i]
    
    def _apply_ridge_acceleration(self):
        """Accelerate wind over ridges."""
        for j in range(self.ny):
            for i in range(self.nx):
                if self.ridge_mask[j, i]:
                    # Accelerate horizontal wind over ridges
                    for k in range(self.nz):
                        altitude = k * self.vertical_resolution
                        terrain_height = self.terrain_elevation_grid[j, i]
                        
                        # Effect strongest just above ridge
                        if terrain_height < altitude < terrain_height + 500:
                            self.wind_u[k, j, i] *= self.ridge_acceleration_factor
                            self.wind_v[k, j, i] *= self.ridge_acceleration_factor
    
    def _apply_thermal_terrain_variation(self):
        """Different terrain types create different thermal patterns."""
        # Only apply during daytime
        hour_of_day = (self.time_of_day + self.time/3600) % 24 if hasattr(self, 'time') else 12
        
        if 10 < hour_of_day < 18:  # Daytime hours
            thermal_base_strength = 2.0 * np.sin(np.pi * (hour_of_day - 10) / 8)
            
            for j in range(self.ny):
                for i in range(self.nx):
                    terrain_type = self.terrain_type_grid[j, i]
                    
                    # Different thermal strengths by terrain type
                    thermal_multiplier = 1.0
                    if terrain_type == TerrainType.ROCK:
                        thermal_multiplier = 1.5  # Rocks heat up more
                    elif terrain_type == TerrainType.GRASS:
                        thermal_multiplier = 1.0  # Normal thermals
                    elif terrain_type == TerrainType.WATER:
                        thermal_multiplier = 0.3  # Water dampens thermals
                    elif terrain_type == TerrainType.SNOW:
                        thermal_multiplier = 0.1  # Snow reflects heat
                    
                    # Apply thermal
                    for k in range(min(6, self.nz)):  # Thermals up to 3km
                        altitude = k * self.vertical_resolution
                        height_factor = max(0, 1.0 - altitude / 3000)
                        
                        thermal_strength = thermal_base_strength * thermal_multiplier * height_factor
                        self.wind_w[k, j, i] += thermal_strength * (0.5 + 0.5 * np.random.randn())
    
    def _apply_lee_turbulence(self):
        """Add turbulence on the lee side of terrain features."""
        # Determine prevailing wind direction
        mean_wind_u = np.mean(self.wind_u[0])
        mean_wind_v = np.mean(self.wind_v[0])
        wind_dir = np.array([mean_wind_u, mean_wind_v])
        wind_dir_norm = wind_dir / (np.linalg.norm(wind_dir) + 1e-6)
        
        for j in range(1, self.ny-1):
            for i in range(1, self.nx-1):
                # Check if this point is in the lee of higher terrain
                # Look upwind
                upwind_i = int(i - wind_dir_norm[0] * 2)
                upwind_j = int(j - wind_dir_norm[1] * 2)
                
                if 0 <= upwind_i < self.nx and 0 <= upwind_j < self.ny:
                    upwind_height = self.terrain_elevation_grid[upwind_j, upwind_i]
                    current_height = self.terrain_elevation_grid[j, i]
                    
                    if upwind_height > current_height + 100:  # In the lee
                        # Add turbulence
                        for k in range(min(4, self.nz)):
                            altitude = k * self.vertical_resolution
                            if altitude < upwind_height + 500:
                                # Increase turbulence
                                if hasattr(self, 'turbulence'):
                                    self.turbulence[k, j, i] *= self.lee_turbulence_factor
    
    def update_with_terrain(self, dt: float):
        """
        Update weather with terrain effects.
        
        Args:
            dt: Time step in seconds
        """
        # Call base update
        self.update(dt)
        
        # Reapply terrain effects after temporal update
        if self.terrain_layer is not None and self.terrain_influence_enabled:
            self._apply_mountain_waves()
            self._apply_valley_channeling()
            self._apply_thermal_terrain_variation()
            
            # Recreate interpolators
            self._create_interpolators()
    
    def get_terrain_influenced_metrics(self, position: np.ndarray) -> Dict[str, Any]:
        """
        Get metrics about terrain influence at a position.
        
        Args:
            position: [x, y, z] position
            
        Returns:
            Dictionary with terrain influence metrics
        """
        if self.terrain_layer is None:
            return {}
        
        x, y, z = position
        
        # Get terrain info at position
        terrain_elevation = self.terrain_layer.get_elevation_at(x, y)
        terrain_type = self.terrain_layer.get_terrain_type_at(x, y)
        gradient = self.terrain_layer.get_gradient(x, y)
        slope = self.terrain_layer.get_slope(x, y)
        
        # Height above ground
        agl = z - terrain_elevation
        
        # Get wind at position
        wind = self.get_wind_at(position)
        
        # Determine if in mountain wave
        in_mountain_wave = agl < 2000 and np.abs(gradient).max() > 0.1
        
        # Determine if in valley
        grid_x = int(x / self.horizontal_resolution)
        grid_y = int(y / self.horizontal_resolution)
        in_valley = False
        if 0 <= grid_x < self.nx and 0 <= grid_y < self.ny:
            in_valley = self.valley_mask[grid_y, grid_x]
        
        return {
            'terrain_elevation': terrain_elevation,
            'agl_altitude': agl,
            'terrain_type': terrain_type.name,
            'terrain_slope': np.degrees(slope),
            'terrain_gradient': gradient,
            'in_mountain_wave': in_mountain_wave,
            'in_valley': in_valley,
            'vertical_wind': wind[2],
            'horizontal_wind_speed': np.linalg.norm(wind[:2])
        }