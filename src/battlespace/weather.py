"""
Weather system for battlespace environment.
Manages wind fields, turbulence, and atmospheric conditions.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.interpolate import RegularGridInterpolator


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
        self.time_of_day = 12.0  # 24-hour format
        self.weather_type = 'clear'  # clear, cloudy, stormy
        
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
            self.wind_interpolator_u(pos),
            self.wind_interpolator_v(pos),
            self.wind_interpolator_w(pos)
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
        Update weather system (for dynamic weather).
        
        Args:
            dt: Time step in seconds
        """
        # Placeholder for dynamic weather updates
        # Could implement:
        # - Moving weather fronts
        # - Time-varying wind patterns
        # - Diurnal thermal cycles
        pass