"""
Asset Manager for centralized aircraft state management.
Integrates with battlespace environment for environmental effects.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import yaml
from pathlib import Path
import time

from rtree import index
from collections import deque

from src.battlespace import Battlespace
from src.assets.aircraft_3dof import Aircraft3DOF, AircraftState, FlightMode


class AssetType(Enum):
    """Types of assets in simulation"""
    INTERCEPTOR = "interceptor"
    TARGET = "target"
    FRIENDLY = "friendly"
    UNKNOWN = "unknown"


@dataclass
class AssetInfo:
    """Metadata about an asset"""
    asset_id: str
    asset_type: AssetType
    aircraft: Aircraft3DOF
    spawn_time: float
    config_file: str
    team: str = "unknown"
    
    # Behavioral parameters
    behavior_mode: str = "manual"  # manual, waypoint, autonomous
    waypoints: List[np.ndarray] = None
    current_waypoint_idx: int = 0
    
    # Tracking
    last_update_time: float = 0.0
    total_distance_traveled: float = 0.0
    previous_position: np.ndarray = None


class AssetManager:
    """
    Central manager for all aircraft assets in the simulation.
    Handles state propagation, environmental integration, and spatial queries.
    """
    
    def __init__(self, battlespace: Battlespace, dt: float = 0.02):
        """
        Initialize asset manager.
        
        Args:
            battlespace: Battlespace environment instance
            dt: Update timestep in seconds (50 Hz default)
        """
        self.battlespace = battlespace
        self.dt = dt
        self.time = 0.0
        
        # Asset storage
        self.assets: Dict[str, AssetInfo] = {}
        self.asset_count = 0
        
        # Spatial index for efficient proximity queries
        self._init_spatial_index()
        
        # History tracking
        self.state_history: Dict[str, deque] = {}
        self.max_history_length = 1000  # Keep last N states per asset
        
        # Performance monitoring
        self.update_times = deque(maxlen=100)
        self.last_update_duration = 0.0
        
        # Collision tracking
        self.collision_pairs: List[Tuple[str, str]] = []
        self.collision_distance_threshold = 50.0  # meters
        
    def _init_spatial_index(self):
        """Initialize R-tree spatial index for efficient queries"""
        p = index.Property()
        p.dimension = 3
        p.variant = index.RT_Star  # R*-tree variant
        self.spatial_idx = index.Index(properties=p)
        self.idx_to_asset_id = {}  # Map index ID to asset ID
        self.next_idx_id = 0
        
    def spawn_aircraft(self, config: Dict[str, Any], asset_id: str, 
                      asset_type: AssetType = AssetType.UNKNOWN,
                      team: str = "unknown") -> str:
        """
        Create and register a new aircraft.
        
        Args:
            config: Aircraft configuration dictionary or path
            asset_id: Unique identifier for the asset
            asset_type: Type of asset (interceptor, target, etc.)
            team: Team affiliation
            
        Returns:
            Asset ID of spawned aircraft
            
        Raises:
            ValueError: If asset_id already exists
        """
        if asset_id in self.assets:
            raise ValueError(f"Asset ID '{asset_id}' already exists")
            
        # Load aircraft configuration
        if isinstance(config, str):
            # It's a file path
            with open(config, 'r') as f:
                loaded_config = yaml.safe_load(f)
            aircraft_config = loaded_config
            config_file = config
        elif isinstance(config, dict):
            # Check if config has an 'aircraft' key that points to a file
            if 'aircraft' in config and isinstance(config['aircraft'], str):
                # Load the aircraft config from file
                with open(config['aircraft'], 'r') as f:
                    loaded_aircraft = yaml.safe_load(f)
                aircraft_config = loaded_aircraft
                config_file = config['aircraft']
            else:
                # Config is already a dictionary with aircraft data
                aircraft_config = config
                config_file = "inline_config"
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
            
        # Create aircraft instance - aircraft_config should have 'aircraft' key
        if 'aircraft' in aircraft_config:
            aircraft = Aircraft3DOF(config_dict=aircraft_config['aircraft'])
        else:
            aircraft = Aircraft3DOF(config_dict=aircraft_config)
        
        # Initialize aircraft state
        initial_state = config.get('initial_state', {})
        position = np.array(initial_state.get('position', [0.0, 0.0, 1000.0]), dtype=np.float64)
        velocity = initial_state.get('velocity', 50.0)
        heading = np.radians(initial_state.get('heading', 0.0))
        climb_angle = np.radians(initial_state.get('climb_angle', 0.0))
        throttle = initial_state.get('throttle', 0.5)
        fuel = initial_state.get('fuel_fraction', 1.0)
        
        # Adjust fuel if given as fraction
        if fuel <= 1.0:
            fuel = fuel * aircraft.fuel_capacity
            
        aircraft.initialize_state(
            position=position,
            velocity=velocity,
            heading=heading,
            flight_path_angle=climb_angle,
            throttle=throttle,
            fuel=fuel
        )
        
        # Create asset info
        asset_info = AssetInfo(
            asset_id=asset_id,
            asset_type=asset_type,
            aircraft=aircraft,
            spawn_time=self.time,
            config_file=config_file,
            team=team,
            previous_position=position.copy()
        )
        
        # Handle waypoints if provided
        if 'waypoints' in config:
            waypoints = [np.array(wp, dtype=np.float64) for wp in config['waypoints']]
            asset_info.waypoints = waypoints
            asset_info.behavior_mode = config.get('behavior', 'waypoint')
            
        # Register asset
        self.assets[asset_id] = asset_info
        self.asset_count += 1
        
        # Add to spatial index
        self._update_spatial_index(asset_id, position)
        
        # Initialize history
        self.state_history[asset_id] = deque(maxlen=self.max_history_length)
        
        print(f"Spawned {asset_type.value} aircraft '{asset_id}' at {position}")
        
        return asset_id
        
    def remove_asset(self, asset_id: str):
        """
        Remove an asset from the simulation.
        
        Args:
            asset_id: ID of asset to remove
        """
        if asset_id not in self.assets:
            return
            
        # Remove from spatial index
        asset = self.assets[asset_id]
        position = asset.aircraft.state.position
        bbox = self._get_bbox(position, 1.0)
        
        # Find and remove from spatial index
        for idx_id in list(self.spatial_idx.intersection(bbox)):
            if self.idx_to_asset_id.get(idx_id) == asset_id:
                self.spatial_idx.delete(idx_id, bbox)
                del self.idx_to_asset_id[idx_id]
                break
                
        # Remove from assets
        del self.assets[asset_id]
        
        # Remove history
        if asset_id in self.state_history:
            del self.state_history[asset_id]
            
        self.asset_count -= 1
        print(f"Removed asset '{asset_id}'")
        
    def update(self):
        """
        Update all assets by one timestep.
        Applies environmental effects from battlespace.
        """
        start_time = time.perf_counter()
        
        # Clear collision pairs from last update
        self.collision_pairs.clear()
        
        # Update each asset
        for asset_id, asset_info in self.assets.items():
            self._update_asset(asset_id, asset_info)
            
        # Check for collisions
        self._check_collisions()
        
        # Update simulation time
        self.time += self.dt
        
        # Track performance
        self.last_update_duration = time.perf_counter() - start_time
        self.update_times.append(self.last_update_duration)
        
    def _update_asset(self, asset_id: str, asset_info: AssetInfo):
        """
        Update a single asset.
        
        Args:
            asset_id: Asset identifier
            asset_info: Asset information
        """
        aircraft = asset_info.aircraft
        
        # Skip if crashed
        if aircraft.mode == FlightMode.CRASHED:
            return
            
        # Get environmental conditions from battlespace
        position = aircraft.state.position
        velocity_vec = aircraft.state.get_velocity_vector()
        
        # Get all environmental effects
        env_effects = self.battlespace.get_aircraft_environment_effects(
            position, velocity_vec
        )
        
        # Extract key environmental parameters
        wind = env_effects['wind_vector']
        air_density = env_effects['air_density']
        terrain_elevation = env_effects['terrain_elevation']
        
        # Apply behavior/control
        if asset_info.behavior_mode == 'waypoint' and asset_info.waypoints:
            self._update_waypoint_behavior(asset_info)
            
        # Update aircraft dynamics
        aircraft.update(self.dt, wind=wind, air_density=air_density)
        
        # Check terrain collision
        agl_altitude = position[2] - terrain_elevation
        if agl_altitude < 10.0:  # Within 10m of terrain
            if aircraft.state.velocity > 20.0:  # Moving fast = crash
                aircraft.mode = FlightMode.CRASHED
                print(f"Asset '{asset_id}' crashed into terrain!")
            else:
                # Soft landing
                aircraft.state.position[2] = terrain_elevation + 1.0
                aircraft.mode = FlightMode.GROUND
                
        # Check airspace violations
        if not self.battlespace.airspace.is_position_valid(position):
            print(f"Warning: Asset '{asset_id}' violated airspace at {position}")
            
        # Update spatial index
        self._update_spatial_index(asset_id, position)
        
        # Track distance traveled
        if asset_info.previous_position is not None:
            distance = np.linalg.norm(position - asset_info.previous_position)
            asset_info.total_distance_traveled += distance
        asset_info.previous_position = position.copy()
        
        # Update timing
        asset_info.last_update_time = self.time
        
        # Store in history
        self._record_state(asset_id, aircraft.state)
        
    def _update_waypoint_behavior(self, asset_info: AssetInfo):
        """
        Update aircraft following waypoint behavior.
        
        Args:
            asset_info: Asset information with waypoints
        """
        if not asset_info.waypoints or asset_info.current_waypoint_idx >= len(asset_info.waypoints):
            return
            
        aircraft = asset_info.aircraft
        current_wp = asset_info.waypoints[asset_info.current_waypoint_idx]
        
        # Calculate distance to waypoint
        position = aircraft.state.position
        distance = np.linalg.norm(current_wp - position)
        
        # Check if waypoint reached (within 50m)
        if distance < 50.0:
            asset_info.current_waypoint_idx += 1
            if asset_info.current_waypoint_idx >= len(asset_info.waypoints):
                # All waypoints complete, loop back or stop
                asset_info.current_waypoint_idx = 0  # Loop for now
                print(f"Asset '{asset_info.asset_id}' completed waypoints, looping")
            return
            
        # Get control commands to fly to waypoint
        bank_cmd, throttle_cmd = aircraft.set_waypoint_commands(
            current_wp, desired_speed=aircraft.v_cruise
        )
        
        # Apply commands
        aircraft.set_controls(bank_angle=bank_cmd, throttle=throttle_cmd)
        
    def _update_spatial_index(self, asset_id: str, position: np.ndarray):
        """
        Update asset position in spatial index.
        
        Args:
            asset_id: Asset identifier
            position: New position
        """
        # Remove old entry if exists
        if asset_id in [self.idx_to_asset_id.get(i) for i in self.idx_to_asset_id]:
            for idx_id, aid in self.idx_to_asset_id.items():
                if aid == asset_id:
                    # Use a large bbox to ensure deletion
                    old_bbox = (-1e6, -1e6, -1e6, 1e6, 1e6, 1e6)
                    try:
                        self.spatial_idx.delete(idx_id, old_bbox)
                    except:
                        pass  # Ignore if not found
                    del self.idx_to_asset_id[idx_id]
                    break
                    
        # Add new entry
        bbox = self._get_bbox(position, 1.0)
        idx_id = self.next_idx_id
        self.next_idx_id += 1
        self.spatial_idx.insert(idx_id, bbox)
        self.idx_to_asset_id[idx_id] = asset_id
        
    def _get_bbox(self, position: np.ndarray, radius: float) -> Tuple[float, ...]:
        """
        Get bounding box for position.
        
        Args:
            position: Center position
            radius: Radius for bbox
            
        Returns:
            Bounding box tuple (min_x, min_y, min_z, max_x, max_y, max_z)
        """
        return (
            position[0] - radius, position[1] - radius, position[2] - radius,
            position[0] + radius, position[1] + radius, position[2] + radius
        )
        
    def _check_collisions(self):
        """Check for collisions between assets"""
        checked_pairs = set()
        
        for asset_id, asset_info in self.assets.items():
            position = asset_info.aircraft.state.position
            
            # Query nearby assets
            search_bbox = self._get_bbox(position, self.collision_distance_threshold)
            nearby_idx_ids = list(self.spatial_idx.intersection(search_bbox))
            
            for idx_id in nearby_idx_ids:
                other_id = self.idx_to_asset_id.get(idx_id)
                if not other_id or other_id == asset_id:
                    continue
                    
                # Skip if already checked this pair
                pair = tuple(sorted([asset_id, other_id]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                
                # Calculate actual distance
                other_position = self.assets[other_id].aircraft.state.position
                distance = np.linalg.norm(position - other_position)
                
                if distance < self.collision_distance_threshold:
                    self.collision_pairs.append(pair)
                    
                    # Very close = collision
                    if distance < 10.0:
                        print(f"COLLISION: '{asset_id}' and '{other_id}' at distance {distance:.1f}m")
                        
    def _record_state(self, asset_id: str, state: AircraftState):
        """
        Record state in history.
        
        Args:
            asset_id: Asset identifier
            state: Current aircraft state
        """
        if asset_id in self.state_history:
            # Store a copy to prevent reference issues
            self.state_history[asset_id].append({
                'time': self.time,
                'position': state.position.copy(),
                'velocity': state.velocity,
                'heading': state.heading,
                'flight_path_angle': state.flight_path_angle,
                'bank_angle': state.bank_angle,
                'throttle': state.throttle,
                'fuel': state.fuel_remaining
            })
            
    def get_asset_state(self, asset_id: str) -> Optional[AircraftState]:
        """
        Get current state of an asset.
        
        Args:
            asset_id: Asset identifier
            
        Returns:
            Aircraft state or None if not found
        """
        if asset_id not in self.assets:
            return None
        return self.assets[asset_id].aircraft.state
        
    def get_assets_in_range(self, position: np.ndarray, range_m: float) -> List[str]:
        """
        Find all assets within range of position.
        
        Args:
            position: Query position [x, y, z]
            range_m: Search radius in meters
            
        Returns:
            List of asset IDs within range
        """
        assets_in_range = []
        search_bbox = self._get_bbox(position, range_m)
        
        for idx_id in self.spatial_idx.intersection(search_bbox):
            asset_id = self.idx_to_asset_id.get(idx_id)
            if not asset_id:
                continue
                
            # Check actual distance
            asset_pos = self.assets[asset_id].aircraft.state.position
            distance = np.linalg.norm(position - asset_pos)
            
            if distance <= range_m:
                assets_in_range.append(asset_id)
                
        return assets_in_range
        
    def get_relative_state(self, from_id: str, to_id: str) -> Optional[Dict[str, Any]]:
        """
        Get relative state between two assets.
        
        Args:
            from_id: Observer asset ID
            to_id: Target asset ID
            
        Returns:
            Dictionary with relative state information
        """
        if from_id not in self.assets or to_id not in self.assets:
            return None
            
        from_aircraft = self.assets[from_id].aircraft
        to_aircraft = self.assets[to_id].aircraft
        
        # Calculate relative position
        from_pos = from_aircraft.state.position
        to_pos = to_aircraft.state.position
        relative_pos = to_pos - from_pos
        
        # Range and bearing
        range_m = np.linalg.norm(relative_pos)
        bearing = np.arctan2(relative_pos[1], relative_pos[0])
        elevation = np.arctan2(relative_pos[2], np.linalg.norm(relative_pos[:2]))
        
        # Relative velocity
        from_vel = from_aircraft.state.get_velocity_vector()
        to_vel = to_aircraft.state.get_velocity_vector()
        relative_vel = to_vel - from_vel
        
        # Closing velocity (positive = closing)
        if range_m > 0:
            closing_velocity = -np.dot(relative_vel, relative_pos) / range_m
        else:
            closing_velocity = 0.0
            
        # Aspect angle (angle off tail of target)
        to_heading = to_aircraft.state.heading
        bearing_from_target = np.arctan2(-relative_pos[1], -relative_pos[0])
        aspect = bearing_from_target - to_heading
        while aspect > np.pi:
            aspect -= 2 * np.pi
        while aspect < -np.pi:
            aspect += 2 * np.pi
            
        return {
            'range': range_m,
            'bearing': bearing,
            'elevation': elevation,
            'relative_position': relative_pos,
            'relative_velocity': relative_vel,
            'closing_velocity': closing_velocity,
            'aspect_angle': aspect,
            'time_to_intercept': range_m / closing_velocity if closing_velocity > 0 else float('inf')
        }
        
    def apply_commands(self, asset_id: str, bank_angle: float, throttle: float):
        """
        Apply control commands to an asset.
        
        Args:
            asset_id: Asset to control
            bank_angle: Commanded bank angle (radians)
            throttle: Commanded throttle [0, 1]
        """
        if asset_id not in self.assets:
            return
            
        aircraft = self.assets[asset_id].aircraft
        aircraft.set_controls(bank_angle=bank_angle, throttle=throttle)
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.update_times:
            return {
                'mean_update_time': 0.0,
                'max_update_time': 0.0,
                'update_rate': 0.0,
                'asset_count': self.asset_count
            }
            
        update_times = list(self.update_times)
        mean_time = np.mean(update_times)
        max_time = np.max(update_times)
        
        return {
            'mean_update_time': mean_time * 1000,  # Convert to ms
            'max_update_time': max_time * 1000,
            'update_rate': 1.0 / mean_time if mean_time > 0 else 0.0,
            'asset_count': self.asset_count,
            'collision_pairs': len(self.collision_pairs),
            'simulation_time': self.time
        }
        
    def get_all_states(self) -> Dict[str, AircraftState]:
        """
        Get states of all assets.
        
        Returns:
            Dictionary mapping asset_id to state
        """
        return {
            asset_id: info.aircraft.state
            for asset_id, info in self.assets.items()
        }
        
    def __str__(self) -> str:
        """String representation"""
        return f"AssetManager: {self.asset_count} assets, t={self.time:.1f}s"