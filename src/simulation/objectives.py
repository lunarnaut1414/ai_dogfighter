"""
Scenario objectives and evaluation system.
Defines and tracks mission objectives for scenario completion.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class ObjectiveType(Enum):
    """Types of scenario objectives"""
    INTERCEPT = "intercept"
    SURVIVE = "survive"
    FUEL_EFFICIENCY = "fuel_efficiency"
    FUEL_REMAINING = "fuel_remaining"  # Alias for fuel_efficiency
    TIME_LIMIT = "time_limit"
    TIME_EFFICIENCY = "time_efficiency"  # Alias for time_limit
    NO_COLLISION = "no_collision"
    NO_TERRAIN_COLLISION = "no_terrain_collision"
    AREA_DENIAL = "area_denial"
    ESCORT = "escort"
    RECONNAISSANCE = "reconnaissance"
    PRIORITIZED_INTERCEPT = "prioritized_intercept"
    ALL_TARGETS_NEUTRALIZED = "all_targets_neutralized"
    MAINTAIN_ALTITUDE = "maintain_altitude"
    REACH_WAYPOINT = "reach_waypoint"


class ObjectiveStatus(Enum):
    """Status of an objective"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ObjectiveResult:
    """Result of an objective evaluation"""
    status: ObjectiveStatus
    progress: float  # 0.0 to 1.0
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterceptEvent:
    """Record of an intercept event"""
    time: float
    interceptor_id: str
    target_id: str
    range: float
    relative_velocity: float
    interceptor_fuel: float
    success: bool


class Objective:
    """Base class for scenario objectives"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize objective from configuration.
        
        Args:
            config: Objective configuration dictionary
        """
        obj_type = config.get('type')
        
        # Handle type aliases
        if obj_type == 'fuel_remaining':
            obj_type = 'fuel_efficiency'
        elif obj_type == 'time_efficiency':
            obj_type = 'time_limit'
        
        # Try to create enum, but handle unknown types gracefully
        try:
            self.type = ObjectiveType(obj_type)
        except (ValueError, KeyError):
            print(f"Warning: Unknown objective type '{obj_type}', using as-is")
            self.type = obj_type
            
        self.description = config.get('description', str(obj_type))
        self.required = config.get('required', True)
        self.weight = config.get('weight', 1.0)
        
        self.status = ObjectiveStatus.PENDING
        self.progress = 0.0
        self.start_time = None
        self.completion_time = None
        self.message = ""
        self.data = {}
        
        # Store raw config for subclasses
        self.config = config
        
    def start(self, current_time: float):
        """Start tracking this objective"""
        self.start_time = current_time
        self.status = ObjectiveStatus.IN_PROGRESS
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """
        Evaluate objective status.
        Must be implemented by subclasses.
        
        Args:
            asset_manager: Asset manager instance
            current_time: Current simulation time
            
        Returns:
            ObjectiveResult with current status
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
        
    def reset(self):
        """Reset objective to initial state"""
        self.status = ObjectiveStatus.PENDING
        self.progress = 0.0
        self.start_time = None
        self.completion_time = None
        self.message = ""
        self.data = {}


class InterceptObjective(Objective):
    """Objective to intercept a specific target"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_id = config.get('target_id', 'any')
        self.intercept_range = config.get('range', 50.0)  # meters
        self.time_limit = config.get('time_limit', float('inf'))
        self.interceptor_id = config.get('interceptor_id', 'interceptor_1')
        
        # Track intercept events
        self.intercept_events = []
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """Check if target has been intercepted"""
        
        # Check time limit
        if self.start_time and (current_time - self.start_time) > self.time_limit:
            return ObjectiveResult(
                status=ObjectiveStatus.FAILED,
                progress=0.0,
                message=f"Time limit exceeded ({self.time_limit}s)",
                data={'time_elapsed': current_time - self.start_time}
            )
            
        # Get interceptor
        interceptor_info = asset_manager.get_asset(self.interceptor_id)
        if not interceptor_info:
            return ObjectiveResult(
                status=ObjectiveStatus.FAILED,
                progress=0.0,
                message=f"Interceptor {self.interceptor_id} not found"
            )
            
        interceptor_pos = interceptor_info['state']['position']
        
        # Check specific target or any target
        if self.target_id == 'any':
            # Find closest target
            all_assets = asset_manager.get_all_assets()
            min_range = float('inf')
            closest_target = None
            
            for asset_id, asset_info in all_assets.items():
                if asset_info['type'] == 'target':
                    target_pos = asset_info['state']['position']
                    range_to_target = np.linalg.norm(target_pos - interceptor_pos)
                    
                    if range_to_target < min_range:
                        min_range = range_to_target
                        closest_target = asset_id
                        
            if closest_target and min_range <= self.intercept_range:
                # Intercept successful
                return ObjectiveResult(
                    status=ObjectiveStatus.COMPLETED,
                    progress=1.0,
                    message=f"Target {closest_target} intercepted at {min_range:.1f}m",
                    data={
                        'target_id': closest_target,
                        'intercept_range': min_range,
                        'time': current_time
                    }
                )
            else:
                # Still pursuing
                progress = max(0.0, 1.0 - (min_range / 1000.0)) if min_range < float('inf') else 0.0
                return ObjectiveResult(
                    status=ObjectiveStatus.IN_PROGRESS,
                    progress=progress,
                    message=f"Pursuing target, range: {min_range:.1f}m" if min_range < float('inf') else "No targets found",
                    data={'min_range': min_range}
                )
        else:
            # Check specific target
            target_info = asset_manager.get_asset(self.target_id)
            
            if not target_info:
                # Target already destroyed/removed
                return ObjectiveResult(
                    status=ObjectiveStatus.COMPLETED,
                    progress=1.0,
                    message=f"Target {self.target_id} neutralized",
                    data={'target_id': self.target_id, 'time': current_time}
                )
                
            target_pos = target_info['state']['position']
            range_to_target = np.linalg.norm(target_pos - interceptor_pos)
            
            if range_to_target <= self.intercept_range:
                # Intercept successful
                return ObjectiveResult(
                    status=ObjectiveStatus.COMPLETED,
                    progress=1.0,
                    message=f"Target {self.target_id} intercepted at {range_to_target:.1f}m",
                    data={
                        'target_id': self.target_id,
                        'intercept_range': range_to_target,
                        'time': current_time
                    }
                )
            else:
                # Still pursuing
                progress = max(0.0, 1.0 - (range_to_target / 1000.0))
                return ObjectiveResult(
                    status=ObjectiveStatus.IN_PROGRESS,
                    progress=progress,
                    message=f"Pursuing {self.target_id}, range: {range_to_target:.1f}m",
                    data={'range': range_to_target}
                )


class FuelEfficiencyObjective(Objective):
    """Objective to maintain fuel above threshold"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_fuel_fraction = config.get('min_fraction', 0.2)
        self.interceptor_id = config.get('interceptor_id', 'interceptor_1')
        self.check_at_end = config.get('check_at_end', True)
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """Check fuel remaining"""
        
        interceptor_info = asset_manager.get_asset(self.interceptor_id)
        if not interceptor_info:
            return ObjectiveResult(
                status=ObjectiveStatus.FAILED,
                progress=0.0,
                message=f"Interceptor {self.interceptor_id} not found"
            )
            
        aircraft = interceptor_info['aircraft']
        fuel_fraction = aircraft.state.fuel_remaining / aircraft.fuel_capacity
        
        if fuel_fraction < self.min_fuel_fraction:
            if not self.check_at_end:
                # Fail immediately if fuel drops below threshold
                return ObjectiveResult(
                    status=ObjectiveStatus.FAILED,
                    progress=fuel_fraction / self.min_fuel_fraction,
                    message=f"Fuel below minimum: {fuel_fraction:.1%} < {self.min_fuel_fraction:.1%}",
                    data={'fuel_fraction': fuel_fraction}
                )
                
        # Still meeting requirement
        progress = min(1.0, fuel_fraction / self.min_fuel_fraction)
        return ObjectiveResult(
            status=ObjectiveStatus.IN_PROGRESS,
            progress=progress,
            message=f"Fuel remaining: {fuel_fraction:.1%}",
            data={'fuel_fraction': fuel_fraction}
        )


class TimeEfficiencyObjective(Objective):
    """Objective to complete mission within time limit"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_time = config.get('max_time', 300.0)  # seconds
        self.time_limit = config.get('time_limit', self.max_time)
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """Check time limit"""
        
        if not self.start_time:
            self.start_time = current_time
            
        elapsed = current_time - self.start_time
        
        if elapsed > self.time_limit:
            return ObjectiveResult(
                status=ObjectiveStatus.FAILED,
                progress=1.0,
                message=f"Time limit exceeded: {elapsed:.1f}s > {self.time_limit:.1f}s",
                data={'elapsed_time': elapsed}
            )
            
        progress = elapsed / self.time_limit
        return ObjectiveResult(
            status=ObjectiveStatus.IN_PROGRESS,
            progress=progress,
            message=f"Time elapsed: {elapsed:.1f}s / {self.time_limit:.1f}s",
            data={'elapsed_time': elapsed, 'remaining_time': self.time_limit - elapsed}
        )


class SurvivalObjective(Objective):
    """Objective for interceptor to survive"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.interceptor_id = config.get('interceptor_id', 'interceptor_1')
        self.duration = config.get('duration', None)  # Survive for specific duration
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """Check if interceptor is still alive"""
        
        interceptor_info = asset_manager.get_asset(self.interceptor_id)
        
        if not interceptor_info:
            return ObjectiveResult(
                status=ObjectiveStatus.FAILED,
                progress=0.0,
                message=f"Interceptor {self.interceptor_id} destroyed",
                data={'time_of_loss': current_time}
            )
            
        # Check if duration requirement met
        if self.duration and self.start_time:
            elapsed = current_time - self.start_time
            if elapsed >= self.duration:
                return ObjectiveResult(
                    status=ObjectiveStatus.COMPLETED,
                    progress=1.0,
                    message=f"Survived for {self.duration}s",
                    data={'survival_time': elapsed}
                )
            else:
                progress = elapsed / self.duration
                return ObjectiveResult(
                    status=ObjectiveStatus.IN_PROGRESS,
                    progress=progress,
                    message=f"Surviving: {elapsed:.1f}s / {self.duration}s",
                    data={'elapsed': elapsed}
                )
        else:
            # Just need to stay alive
            return ObjectiveResult(
                status=ObjectiveStatus.IN_PROGRESS,
                progress=1.0,
                message="Interceptor operational",
                data={'time': current_time}
            )


class NoCollisionObjective(Objective):
    """Objective to avoid collisions"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.interceptor_id = config.get('interceptor_id', 'interceptor_1')
        self.min_separation = config.get('min_separation', 50.0)  # meters
        self.check_terrain = config.get('check_terrain', True)
        self.terrain_margin = config.get('terrain_margin', 100.0)  # meters above terrain
        
        # Track near-misses
        self.near_misses = []
        self.collisions = []
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """Check for collisions or near-misses"""
        
        interceptor_info = asset_manager.get_asset(self.interceptor_id)
        if not interceptor_info:
            return ObjectiveResult(
                status=ObjectiveStatus.FAILED,
                progress=0.0,
                message="Interceptor lost"
            )
            
        interceptor_pos = interceptor_info['state']['position']
        
        # Check terrain collision
        if self.check_terrain:
            altitude = interceptor_pos[2]
            
            # Get terrain height (simplified - should query battlespace)
            terrain_height = 0.0  # Would get from battlespace.get_terrain_height()
            
            height_above_terrain = altitude - terrain_height
            
            if height_above_terrain < 0:
                self.collisions.append({
                    'type': 'terrain',
                    'time': current_time,
                    'position': interceptor_pos.copy()
                })
                return ObjectiveResult(
                    status=ObjectiveStatus.FAILED,
                    progress=0.0,
                    message="Terrain collision",
                    data={'collision_type': 'terrain', 'position': interceptor_pos}
                )
            elif height_above_terrain < self.terrain_margin:
                # Near terrain - warning
                self.near_misses.append({
                    'type': 'terrain',
                    'time': current_time,
                    'clearance': height_above_terrain
                })
                
        # Check aircraft collisions
        all_assets = asset_manager.get_all_assets()
        min_distance = float('inf')
        closest_aircraft = None
        
        for asset_id, asset_info in all_assets.items():
            if asset_id == self.interceptor_id:
                continue
                
            other_pos = asset_info['state']['position']
            distance = np.linalg.norm(other_pos - interceptor_pos)
            
            if distance < min_distance:
                min_distance = distance
                closest_aircraft = asset_id
                
            if distance < self.min_separation:
                # Near-miss or collision
                if distance < 10.0:  # Actual collision threshold
                    self.collisions.append({
                        'type': 'aircraft',
                        'other_id': asset_id,
                        'time': current_time,
                        'distance': distance
                    })
                    return ObjectiveResult(
                        status=ObjectiveStatus.FAILED,
                        progress=0.0,
                        message=f"Collision with {asset_id}",
                        data={'collision_with': asset_id, 'distance': distance}
                    )
                else:
                    self.near_misses.append({
                        'type': 'aircraft',
                        'other_id': asset_id,
                        'time': current_time,
                        'distance': distance
                    })
                    
        # No collisions
        safety_margin = min_distance / self.min_separation if min_distance < float('inf') else 1.0
        return ObjectiveResult(
            status=ObjectiveStatus.IN_PROGRESS,
            progress=min(1.0, safety_margin),
            message=f"Clear, min separation: {min_distance:.1f}m",
            data={
                'min_distance': min_distance,
                'closest_aircraft': closest_aircraft,
                'near_misses': len(self.near_misses)
            }
        )


class ReachWaypointObjective(Objective):
    """Objective to reach specific waypoint(s)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.waypoints = [np.array(wp) for wp in config.get('waypoints', [])]
        self.tolerance = config.get('tolerance', 100.0)  # meters
        self.interceptor_id = config.get('interceptor_id', 'interceptor_1')
        self.require_all = config.get('require_all', True)
        
        self.current_waypoint_idx = 0
        self.reached_waypoints = []
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """Check if waypoints have been reached"""
        
        if not self.waypoints:
            return ObjectiveResult(
                status=ObjectiveStatus.COMPLETED,
                progress=1.0,
                message="No waypoints defined"
            )
            
        interceptor_info = asset_manager.get_asset(self.interceptor_id)
        if not interceptor_info:
            return ObjectiveResult(
                status=ObjectiveStatus.FAILED,
                progress=0.0,
                message="Interceptor not found"
            )
            
        interceptor_pos = interceptor_info['state']['position']
        
        # Check current waypoint
        if self.current_waypoint_idx < len(self.waypoints):
            current_wp = self.waypoints[self.current_waypoint_idx]
            distance = np.linalg.norm(interceptor_pos - current_wp)
            
            if distance <= self.tolerance:
                # Waypoint reached
                self.reached_waypoints.append(self.current_waypoint_idx)
                self.current_waypoint_idx += 1
                
                if self.current_waypoint_idx >= len(self.waypoints):
                    # All waypoints reached
                    return ObjectiveResult(
                        status=ObjectiveStatus.COMPLETED,
                        progress=1.0,
                        message=f"All {len(self.waypoints)} waypoints reached",
                        data={'waypoints_reached': len(self.reached_waypoints)}
                    )
                    
        # Still in progress
        if self.current_waypoint_idx < len(self.waypoints):
            current_wp = self.waypoints[self.current_waypoint_idx]
            distance = np.linalg.norm(interceptor_pos - current_wp)
            
            progress = len(self.reached_waypoints) / len(self.waypoints)
            # Add partial progress for current waypoint
            progress += (1.0 - min(1.0, distance / 1000.0)) / len(self.waypoints)
            
            return ObjectiveResult(
                status=ObjectiveStatus.IN_PROGRESS,
                progress=progress,
                message=f"Waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)}, "
                        f"distance: {distance:.1f}m",
                data={
                    'current_waypoint': self.current_waypoint_idx,
                    'distance_to_waypoint': distance,
                    'waypoints_reached': len(self.reached_waypoints)
                }
            )
        else:
            # All waypoints reached
            return ObjectiveResult(
                status=ObjectiveStatus.COMPLETED,
                progress=1.0,
                message=f"All waypoints reached",
                data={'waypoints_reached': len(self.reached_waypoints)}
            )


class MaintainAltitudeObjective(Objective):
    """Objective to maintain altitude within bounds"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_altitude = config.get('min_altitude', 500.0)  # meters
        self.max_altitude = config.get('max_altitude', 5000.0)  # meters
        self.interceptor_id = config.get('interceptor_id', 'interceptor_1')
        
        # Track violations
        self.violations = []
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """Check altitude constraints"""
        
        interceptor_info = asset_manager.get_asset(self.interceptor_id)
        if not interceptor_info:
            return ObjectiveResult(
                status=ObjectiveStatus.FAILED,
                progress=0.0,
                message="Interceptor not found"
            )
            
        altitude = interceptor_info['state']['position'][2]
        
        if altitude < self.min_altitude:
            self.violations.append({
                'type': 'too_low',
                'altitude': altitude,
                'time': current_time
            })
            return ObjectiveResult(
                status=ObjectiveStatus.FAILED,
                progress=0.0,
                message=f"Altitude too low: {altitude:.1f}m < {self.min_altitude:.1f}m",
                data={'altitude': altitude, 'violation': 'too_low'}
            )
        elif altitude > self.max_altitude:
            self.violations.append({
                'type': 'too_high',
                'altitude': altitude,
                'time': current_time
            })
            return ObjectiveResult(
                status=ObjectiveStatus.FAILED,
                progress=0.0,
                message=f"Altitude too high: {altitude:.1f}m > {self.max_altitude:.1f}m",
                data={'altitude': altitude, 'violation': 'too_high'}
            )
        else:
            # Within bounds - calculate how centered we are
            altitude_range = self.max_altitude - self.min_altitude
            center = (self.max_altitude + self.min_altitude) / 2
            deviation = abs(altitude - center) / (altitude_range / 2)
            progress = 1.0 - deviation
            
            return ObjectiveResult(
                status=ObjectiveStatus.IN_PROGRESS,
                progress=progress,
                message=f"Altitude OK: {altitude:.1f}m",
                data={
                    'altitude': altitude,
                    'margin_low': altitude - self.min_altitude,
                    'margin_high': self.max_altitude - altitude
                }
            )