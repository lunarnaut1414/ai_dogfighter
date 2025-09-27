"""
Integrated simulation environment.
Orchestrates battlespace, assets, and simulation components.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time

from ..battlespace.core import Battlespace
from ..assets.asset_manager import AssetManager, AssetType
from ..assets.sensor_model import SensorModel
from .dynamics import DynamicsModel
from .sensors import SensorSuite


class SimulationEnvironment:
    """
    Main simulation environment that integrates all components.
    Provides unified interface for scenario execution.
    """
    
    def __init__(self, 
                 battlespace_config: Optional[str] = None,
                 dt: float = 0.02,
                 enable_physics: bool = True,
                 enable_sensors: bool = True,
                 enable_weather: bool = True):
        """
        Initialize simulation environment.
        
        Args:
            battlespace_config: Path to battlespace configuration file
            dt: Simulation timestep in seconds
            enable_physics: Enable physics simulation
            enable_sensors: Enable sensor simulation
            enable_weather: Enable weather effects
        """
        self.dt = dt
        self.time = 0.0
        self.step_count = 0
        
        # Feature flags
        self.physics_enabled = enable_physics
        self.sensors_enabled = enable_sensors
        self.weather_enabled = enable_weather
        
        # Initialize battlespace
        if battlespace_config:
            self.battlespace = Battlespace(config_file=battlespace_config)
            self.battlespace.generate()
        else:
            self.battlespace = Battlespace()
            self.battlespace.generate()
            
        # Initialize asset manager
        self.asset_manager = AssetManager(self.battlespace, dt=dt)
        
        # Initialize sensor suite
        if enable_sensors:
            self.sensor_suite = SensorSuite()
            
        # Dynamics model for physics calculations
        if enable_physics:
            self.dynamics = DynamicsModel()
            
        # Performance tracking
        self.last_step_time = 0.0
        self.step_times = []
        
        # Data recording
        self.history = {
            'time': [],
            'states': [],
            'events': []
        }
        
    def step(self) -> Dict[str, Any]:
        """
        Execute one simulation timestep.
        
        Returns:
            Dictionary with step results and metrics
        """
        step_start = time.perf_counter()
        
        # Update weather if enabled
        if self.weather_enabled:
            self.battlespace.weather.update(self.dt)
            
        # Get all assets
        assets = self.asset_manager.get_all_assets()
        
        # Process each asset
        for asset_id, asset in assets.items():
            if asset['type'] in [AssetType.INTERCEPTOR, AssetType.TARGET]:
                # Get current state
                state = asset['state']
                position = state['position']
                
                # Apply environmental effects
                env_effects = self._get_environmental_effects(position, state)
                
                # Apply effects to asset
                if 'aircraft' in asset:
                    aircraft = asset['aircraft']
                    
                    # Apply wind
                    if self.weather_enabled and env_effects['wind'] is not None:
                        aircraft.apply_wind(env_effects['wind'])
                        
                    # Apply atmospheric effects
                    if env_effects['density_ratio'] != 1.0:
                        aircraft.air_density = env_effects['air_density']
                        
                    # Check terrain constraints
                    min_alt = env_effects['min_safe_altitude']
                    if position[2] < min_alt + 50:  # 50m safety margin
                        # Terrain avoidance
                        aircraft.commanded_altitude = min_alt + 100
                        
        # Update asset manager (propagates all aircraft)
        self.asset_manager.update()
        
        # Update sensors if enabled
        sensor_data = {}
        if self.sensors_enabled:
            sensor_data = self._update_sensors()
            
        # Record history
        self._record_state()
        
        # Update time
        self.time += self.dt
        self.step_count += 1
        
        # Track performance
        step_time = time.perf_counter() - step_start
        self.last_step_time = step_time
        self.step_times.append(step_time)
        
        # Compile step results
        return {
            'time': self.time,
            'step': self.step_count,
            'assets': len(assets),
            'sensor_data': sensor_data,
            'step_time_ms': step_time * 1000,
            'realtime_factor': self.dt / step_time if step_time > 0 else 0
        }
        
    def _get_environmental_effects(self, position: np.ndarray, 
                                  state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get all environmental effects at a position.
        
        Args:
            position: 3D position [x, y, z]
            state: Aircraft state dictionary
            
        Returns:
            Dictionary of environmental effects
        """
        effects = {
            'wind': None,
            'turbulence': 0.0,
            'air_density': 1.225,
            'density_ratio': 1.0,
            'temperature': 15.0,
            'pressure': 101325.0,
            'terrain_elevation': 0.0,
            'min_safe_altitude': 0.0,
            'visibility': 50000.0
        }
        
        # Get terrain elevation
        effects['terrain_elevation'] = self.battlespace.terrain.get_elevation(
            position[0], position[1]
        )
        
        # Calculate minimum safe altitude
        effects['min_safe_altitude'] = self.battlespace.get_minimum_safe_altitude(
            position[0], position[1], 
            radius=500, safety_margin=50
        )
        
        # Get weather effects
        if self.weather_enabled:
            # Wind at position
            effects['wind'] = self.battlespace.weather.get_wind_at_position(position)
            
            # Turbulence
            effects['turbulence'] = self.battlespace.weather.get_turbulence_intensity(
                position
            )
            
            # Atmospheric conditions
            atmos = self.battlespace.weather.get_atmospheric_conditions(position[2])
            effects['temperature'] = atmos['temperature']
            effects['pressure'] = atmos['pressure']
            effects['air_density'] = atmos['density']
            effects['density_ratio'] = atmos['density'] / 1.225
            
            # Visibility
            effects['visibility'] = self.battlespace.weather.get_visibility()
            
        return effects
        
    def _update_sensors(self) -> Dict[str, Any]:
        """
        Update all sensor models.
        
        Returns:
            Dictionary of sensor detections
        """
        sensor_data = {}
        
        # Get interceptor
        interceptor = None
        for asset_id, asset in self.asset_manager.assets.items():
            if asset['type'] == AssetType.INTERCEPTOR:
                interceptor = asset
                break
                
        if not interceptor:
            return sensor_data
            
        # Detect all targets
        detections = []
        interceptor_pos = interceptor['state']['position']
        
        for asset_id, asset in self.asset_manager.assets.items():
            if asset['type'] == AssetType.TARGET:
                target_pos = asset['state']['position']
                
                # Check line of sight
                has_los = self.battlespace.get_line_of_sight(
                    interceptor_pos, target_pos
                )
                
                if has_los:
                    # Calculate range
                    range_to_target = np.linalg.norm(
                        np.array(target_pos) - np.array(interceptor_pos)
                    )
                    
                    # Create detection
                    detection = {
                        'id': asset_id,
                        'position': target_pos,
                        'range': range_to_target,
                        'bearing': np.arctan2(
                            target_pos[1] - interceptor_pos[1],
                            target_pos[0] - interceptor_pos[0]
                        ),
                        'elevation': np.arctan2(
                            target_pos[2] - interceptor_pos[2],
                            range_to_target
                        ),
                        'time': self.time
                    }
                    
                    detections.append(detection)
                    
        sensor_data['detections'] = detections
        sensor_data['detection_count'] = len(detections)
        
        return sensor_data
        
    def _record_state(self):
        """Record current simulation state for analysis."""
        # Limit history size
        max_history = 10000
        if len(self.history['time']) >= max_history:
            # Remove oldest entries
            for key in self.history:
                if isinstance(self.history[key], list):
                    self.history[key] = self.history[key][-max_history//2:]
                    
        # Record current state
        self.history['time'].append(self.time)
        
        # Record asset states
        states = {}
        for asset_id, asset in self.asset_manager.assets.items():
            states[asset_id] = {
                'position': asset['state']['position'].tolist(),
                'velocity': asset['state'].get('velocity', 0),
                'heading': asset['state'].get('heading', 0),
                'altitude': asset['state']['position'][2]
            }
        self.history['states'].append(states)
        
    def spawn_aircraft(self, 
                      aircraft_type: str,
                      position: List[float],
                      heading: float = 0.0,
                      velocity: float = 50.0,
                      team: str = 'neutral') -> str:
        """
        Spawn an aircraft in the environment.
        
        Args:
            aircraft_type: Type of aircraft ('interceptor' or 'target')
            position: Initial position [x, y, z]
            heading: Initial heading in degrees
            velocity: Initial velocity in m/s
            team: Team affiliation
            
        Returns:
            Aircraft ID
        """
        # Determine asset type
        if aircraft_type == 'interceptor':
            asset_type = AssetType.INTERCEPTOR
            config_file = 'configs/aircraft/interceptor_drone.yaml'
        else:
            asset_type = AssetType.TARGET
            config_file = 'configs/aircraft/target_basic.yaml'
            
        # Create spawn configuration
        config = {
            'aircraft_config': config_file,
            'initial_state': {
                'position': position,
                'velocity': velocity,
                'heading': np.radians(heading),
                'throttle': 0.7
            }
        }
        
        # Spawn through asset manager
        asset_id = f"{aircraft_type}_{self.step_count}"
        return self.asset_manager.spawn_aircraft(
            config=config,
            asset_id=asset_id,
            asset_type=asset_type,
            team=team
        )
        
    def remove_aircraft(self, asset_id: str) -> bool:
        """
        Remove an aircraft from the environment.
        
        Args:
            asset_id: Aircraft ID to remove
            
        Returns:
            True if removed successfully
        """
        return self.asset_manager.remove_asset(asset_id)
        
    def set_aircraft_target(self, interceptor_id: str, target_id: str):
        """
        Set target for an interceptor.
        
        Args:
            interceptor_id: Interceptor aircraft ID
            target_id: Target aircraft ID
        """
        interceptor = self.asset_manager.get_asset(interceptor_id)
        if interceptor and interceptor['type'] == AssetType.INTERCEPTOR:
            interceptor['current_target'] = target_id
            
    def get_state_vector(self) -> np.ndarray:
        """
        Get full state vector for all assets.
        Useful for reinforcement learning or analysis.
        
        Returns:
            Numpy array of all asset states
        """
        states = []
        
        for asset_id, asset in self.asset_manager.assets.items():
            state = asset['state']
            states.extend([
                state['position'][0],
                state['position'][1], 
                state['position'][2],
                state.get('velocity', 0),
                state.get('heading', 0),
                state.get('climb_angle', 0)
            ])
            
        return np.array(states)
        
    def reset(self):
        """Reset environment to initial state."""
        self.time = 0.0
        self.step_count = 0
        self.asset_manager.reset()
        self.history = {
            'time': [],
            'states': [],
            'events': []
        }
        self.step_times = []
        
        # Regenerate battlespace with new seed
        self.battlespace.generate()
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get simulation metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            'simulation_time': self.time,
            'steps': self.step_count,
            'assets': len(self.asset_manager.assets),
            'update_rate': 1.0 / self.dt
        }
        
        if self.step_times:
            metrics['mean_step_time_ms'] = np.mean(self.step_times) * 1000
            metrics['max_step_time_ms'] = np.max(self.step_times) * 1000
            metrics['realtime_factor'] = self.dt / np.mean(self.step_times)
            
        return metrics
        
    def close(self):
        """Clean up resources."""
        pass