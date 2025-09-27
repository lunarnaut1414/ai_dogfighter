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
    AREA_DENIAL = "area_denial"
    ESCORT = "escort"
    RECONNAISSANCE = "reconnaissance"
    PRIORITIZED_INTERCEPT = "prioritized_intercept"
    ALL_TARGETS_NEUTRALIZED = "all_targets_neutralized"


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
        except ValueError:
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
        
    def start(self, current_time: float):
        """Start tracking this objective"""
        self.start_time = current_time
        self.status = ObjectiveStatus.IN_PROGRESS
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """
        Evaluate objective status.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
        
    def complete(self, current_time: float, message: str = ""):
        """Mark objective as completed"""
        self.completion_time = current_time
        self.status = ObjectiveStatus.COMPLETED
        self.progress = 1.0
        self.message = message or f"{self.description} completed"
        
    def fail(self, current_time: float, message: str = ""):
        """Mark objective as failed"""
        self.completion_time = current_time
        self.status = ObjectiveStatus.FAILED
        self.message = message or f"{self.description} failed"


class InterceptObjective(Objective):
    """Intercept a specific target within range"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_id = config.get('target_id', config.get('target'))
        self.intercept_range = config.get('range', 100.0)  # meters
        self.time_limit = config.get('time_limit', None)  # seconds
        self.intercept_events = []
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """Check if target has been intercepted"""
        
        # Check time limit
        if self.time_limit and current_time - self.start_time > self.time_limit:
            self.fail(current_time, f"Time limit exceeded ({self.time_limit}s)")
            return ObjectiveResult(ObjectiveStatus.FAILED, 0.0, self.message)
            
        # Get interceptor and target
        interceptor = None
        target = None
        
        for asset_id in asset_manager.assets:
            # Use get_asset method if available, otherwise handle AssetInfo directly
            if hasattr(asset_manager, 'get_asset'):
                asset = asset_manager.get_asset(asset_id)
                if asset and asset['type'] == 'interceptor':
                    interceptor = asset
                elif asset and asset_id == self.target_id:
                    target = asset
            else:
                # Direct AssetInfo access
                asset_info = asset_manager.assets[asset_id]
                if asset_info.asset_type.value == 'interceptor':
                    interceptor = {
                        'state': {
                            'position': asset_info.aircraft.state.position,
                            'fuel_fraction': asset_info.aircraft.state.fuel_remaining / asset_info.aircraft.fuel_capacity
                        }
                    }
                elif asset_id == self.target_id:
                    target = {
                        'state': {
                            'position': asset_info.aircraft.state.position
                        }
                    }
                
        if not interceptor or not target:
            return ObjectiveResult(ObjectiveStatus.IN_PROGRESS, 0.0, 
                                 "Waiting for assets")
            
        # Calculate range
        interceptor_pos = np.array(interceptor['state']['position'])
        target_pos = np.array(target['state']['position'])
        range_to_target = np.linalg.norm(target_pos - interceptor_pos)
        
        # Update progress based on closing distance
        initial_range = self.data.get('initial_range')
        if initial_range is None:
            initial_range = range_to_target
            self.data['initial_range'] = initial_range
            
        if initial_range > self.intercept_range:
            self.progress = max(0, 1.0 - (range_to_target - self.intercept_range) / 
                              (initial_range - self.intercept_range))
        else:
            self.progress = 1.0
            
        # Check intercept condition
        if range_to_target <= self.intercept_range:
            # Record intercept event
            event = InterceptEvent(
                time=current_time,
                interceptor_id=interceptor['id'],
                target_id=self.target_id,
                range=range_to_target,
                relative_velocity=0,  # TODO: Calculate
                interceptor_fuel=interceptor['state'].get('fuel_fraction', 1.0),
                success=True
            )
            self.intercept_events.append(event)
            
            self.complete(current_time, 
                         f"Target {self.target_id} intercepted at {range_to_target:.1f}m")
            return ObjectiveResult(ObjectiveStatus.COMPLETED, 1.0, self.message, 
                                 {'intercept_event': event})
            
        return ObjectiveResult(ObjectiveStatus.IN_PROGRESS, self.progress,
                             f"Range to target: {range_to_target:.1f}m")


class AllTargetsNeutralizedObjective(Objective):
    """Neutralize all hostile targets"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.intercept_range = config.get('range', 100.0)
        self.time_limit = config.get('time_limit', None)
        self.neutralized_targets = set()
        self.total_targets = None
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """Check if all targets have been neutralized"""
        
        # Check time limit
        if self.time_limit and current_time - self.start_time > self.time_limit:
            self.fail(current_time, f"Time limit exceeded")
            return ObjectiveResult(ObjectiveStatus.FAILED, self.progress, self.message)
            
        # Find all hostile targets
        hostile_targets = []
        interceptor = None
        
        for asset_id in asset_manager.assets:
            # Use get_asset method if available
            if hasattr(asset_manager, 'get_asset'):
                asset = asset_manager.get_asset(asset_id)
                if asset:
                    if asset['type'] == 'interceptor':
                        interceptor = asset
                    elif asset['type'] == 'target' and asset.get('team') != 'blue':
                        hostile_targets.append(asset)
            else:
                # Direct AssetInfo access
                asset_info = asset_manager.assets[asset_id]
                if asset_info.asset_type.value == 'interceptor':
                    interceptor = {
                        'state': {
                            'position': asset_info.aircraft.state.position
                        },
                        'id': asset_info.asset_id
                    }
                elif asset_info.asset_type.value == 'target' and asset_info.team != 'blue':
                    hostile_targets.append({
                        'state': {
                            'position': asset_info.aircraft.state.position
                        },
                        'id': asset_info.asset_id
                    })
                
        if self.total_targets is None:
            self.total_targets = len(hostile_targets)
            
        if not interceptor:
            return ObjectiveResult(ObjectiveStatus.IN_PROGRESS, 0.0, "No interceptor")
            
        # Check each target
        interceptor_pos = np.array(interceptor['state']['position'])
        
        for target in hostile_targets:
            target_id = target['id']
            if target_id not in self.neutralized_targets:
                target_pos = np.array(target['state']['position'])
                range_to_target = np.linalg.norm(target_pos - interceptor_pos)
                
                if range_to_target <= self.intercept_range:
                    self.neutralized_targets.add(target_id)
                    
        # Update progress
        if self.total_targets > 0:
            self.progress = len(self.neutralized_targets) / self.total_targets
        else:
            self.progress = 1.0
            
        # Check completion
        if len(self.neutralized_targets) >= self.total_targets:
            self.complete(current_time, 
                         f"All {self.total_targets} targets neutralized")
            return ObjectiveResult(ObjectiveStatus.COMPLETED, 1.0, self.message)
            
        return ObjectiveResult(ObjectiveStatus.IN_PROGRESS, self.progress,
                             f"Neutralized {len(self.neutralized_targets)}/{self.total_targets}")


class PrioritizedInterceptObjective(Objective):
    """Intercept targets in priority order"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_order = config.get('order', [])
        self.intercept_range = config.get('range', 100.0)
        self.current_target_index = 0
        self.intercepted_targets = []
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """Check intercept progress following priority order"""
        
        if self.current_target_index >= len(self.target_order):
            self.complete(current_time, "All priority targets intercepted")
            return ObjectiveResult(ObjectiveStatus.COMPLETED, 1.0, self.message)
            
        current_target = self.target_order[self.current_target_index]
        
        # Get interceptor and target
        interceptor = None
        target = None
        
        for asset_id, asset in asset_manager.assets.items():
            if asset['type'] == 'interceptor':
                interceptor = asset
            elif asset_id == current_target:
                target = asset
                
        if not interceptor or not target:
            return ObjectiveResult(ObjectiveStatus.IN_PROGRESS, self.progress,
                                 f"Seeking target {current_target}")
            
        # Check range to current target
        interceptor_pos = np.array(interceptor['state']['position'])
        target_pos = np.array(target['state']['position'])
        range_to_target = np.linalg.norm(target_pos - interceptor_pos)
        
        if range_to_target <= self.intercept_range:
            self.intercepted_targets.append(current_target)
            self.current_target_index += 1
            self.progress = self.current_target_index / len(self.target_order)
            
        return ObjectiveResult(ObjectiveStatus.IN_PROGRESS, self.progress,
                             f"Target {self.current_target_index + 1}/{len(self.target_order)}: "
                             f"{current_target} at {range_to_target:.1f}m")


class FuelEfficiencyObjective(Objective):
    """Maintain minimum fuel reserves"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_fuel_fraction = config.get('min_remaining', 0.2)
        self.min_fraction = config.get('min_fraction', 0.2)  # Alternative key
        if self.min_fraction != 0.2:
            self.min_fuel_fraction = self.min_fraction
            
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """Check fuel efficiency"""
        
        # Find interceptor
        interceptor = None
        for asset_id, asset in asset_manager.assets.items():
            if asset['type'] == 'interceptor':
                interceptor = asset
                break
                
        if not interceptor:
            return ObjectiveResult(ObjectiveStatus.IN_PROGRESS, 0.0, "No interceptor")
            
        fuel_fraction = interceptor['state'].get('fuel_fraction', 1.0)
        self.progress = min(1.0, fuel_fraction / self.min_fuel_fraction)
        
        if fuel_fraction < self.min_fuel_fraction:
            self.fail(current_time, 
                     f"Fuel below minimum ({fuel_fraction:.1%} < {self.min_fuel_fraction:.1%})")
            return ObjectiveResult(ObjectiveStatus.FAILED, self.progress, self.message)
            
        return ObjectiveResult(ObjectiveStatus.IN_PROGRESS, self.progress,
                             f"Fuel: {fuel_fraction:.1%}")


class TimeLimitObjective(Objective):
    """Complete mission within time limit"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_time = config.get('max_time', 300.0)
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """Check time limit"""
        
        elapsed = current_time - self.start_time
        self.progress = min(1.0, elapsed / self.max_time)
        
        if elapsed >= self.max_time:
            self.fail(current_time, f"Time limit exceeded ({self.max_time}s)")
            return ObjectiveResult(ObjectiveStatus.FAILED, 1.0, self.message)
            
        return ObjectiveResult(ObjectiveStatus.IN_PROGRESS, self.progress,
                             f"Time: {elapsed:.1f}/{self.max_time}s")


class NoCollisionObjective(Objective):
    """Avoid terrain and aircraft collisions"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_altitude = config.get('min_altitude', 50.0)
        self.min_separation = config.get('min_separation', 50.0)
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        """Check for collisions"""
        
        # Check terrain collisions
        for asset_id, asset in asset_manager.assets.items():
            pos = asset['state']['position']
            
            # Check altitude
            if pos[2] < self.min_altitude:
                self.fail(current_time, 
                         f"{asset_id} below minimum altitude ({pos[2]:.1f}m)")
                return ObjectiveResult(ObjectiveStatus.FAILED, 0.0, self.message)
                
        # Check aircraft separation
        assets = list(asset_manager.assets.values())
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                pos1 = np.array(assets[i]['state']['position'])
                pos2 = np.array(assets[j]['state']['position'])
                separation = np.linalg.norm(pos1 - pos2)
                
                if separation < self.min_separation:
                    self.fail(current_time,
                             f"Collision between {assets[i]['id']} and {assets[j]['id']}")
                    return ObjectiveResult(ObjectiveStatus.FAILED, 0.0, self.message)
                    
        self.progress = 1.0  # Binary objective
        return ObjectiveResult(ObjectiveStatus.IN_PROGRESS, 1.0, "No collisions")


class ObjectiveManager:
    """Manages all objectives for a scenario"""
    
    def __init__(self):
        self.objectives: List[Objective] = []
        self.primary_objectives: List[Objective] = []
        self.secondary_objectives: List[Objective] = []
        self.completed_objectives: List[Objective] = []
        self.failed_objectives: List[Objective] = []
        
    def add_objective(self, config: Dict[str, Any], primary: bool = True):
        """
        Add an objective from configuration.
        
        Args:
            config: Objective configuration
            primary: Whether this is a primary objective
        """
        obj_type = config.get('type')
        
        # Handle type aliases
        if obj_type == 'fuel_remaining':
            obj_type = 'fuel_efficiency'
        elif obj_type == 'time_efficiency':
            obj_type = 'time_limit'
        
        # Create appropriate objective instance
        if obj_type == 'intercept':
            obj = InterceptObjective(config)
        elif obj_type == 'all_targets_neutralized':
            obj = AllTargetsNeutralizedObjective(config)
        elif obj_type == 'prioritized_intercept':
            obj = PrioritizedInterceptObjective(config)
        elif obj_type in ['fuel_efficiency', 'fuel_remaining']:
            obj = FuelEfficiencyObjective(config)
        elif obj_type in ['time_efficiency', 'time_limit']:
            obj = TimeLimitObjective(config)
        elif obj_type == 'no_collision':
            obj = NoCollisionObjective(config)
        else:
            # Skip unknown objective types with warning
            print(f"Warning: Unknown objective type '{obj_type}', skipping")
            return
            
        self.objectives.append(obj)
        
        if primary:
            self.primary_objectives.append(obj)
        else:
            self.secondary_objectives.append(obj)
            
    def start_all(self, current_time: float):
        """Start tracking all objectives"""
        for obj in self.objectives:
            obj.start(current_time)
            
    def evaluate_all(self, asset_manager, current_time: float) -> Dict[str, Any]:
        """
        Evaluate all objectives and return status.
        
        Returns:
            Dictionary with overall status and individual objective results
        """
        results = {
            'primary': [],
            'secondary': [],
            'all_completed': False,
            'primary_completed': False,
            'any_failed': False,
            'overall_progress': 0.0
        }
        
        total_weight = 0.0
        weighted_progress = 0.0
        
        for obj in self.objectives:
            if obj.status in [ObjectiveStatus.COMPLETED, ObjectiveStatus.FAILED]:
                continue
                
            result = obj.evaluate(asset_manager, current_time)
            
            # Update objective state
            if result.status == ObjectiveStatus.COMPLETED:
                obj.status = ObjectiveStatus.COMPLETED
                obj.progress = 1.0
                self.completed_objectives.append(obj)
            elif result.status == ObjectiveStatus.FAILED:
                obj.status = ObjectiveStatus.FAILED
                self.failed_objectives.append(obj)
                if obj.required:
                    results['any_failed'] = True
                    
            # Track progress
            total_weight += obj.weight
            weighted_progress += obj.progress * obj.weight
            
            # Add to results
            obj_result = {
                'description': obj.description,
                'status': obj.status.value,
                'progress': obj.progress,
                'required': obj.required,
                'message': result.message
            }
            
            if obj in self.primary_objectives:
                results['primary'].append(obj_result)
            else:
                results['secondary'].append(obj_result)
                
        # Calculate overall progress
        if total_weight > 0:
            results['overall_progress'] = weighted_progress / total_weight
            
        # Check completion
        primary_complete = all(obj.status == ObjectiveStatus.COMPLETED 
                              for obj in self.primary_objectives if obj.required)
        all_complete = all(obj.status == ObjectiveStatus.COMPLETED 
                          for obj in self.objectives if obj.required)
                          
        results['primary_completed'] = primary_complete
        results['all_completed'] = all_complete
        
        return results
        
    def get_intercept_events(self) -> List[InterceptEvent]:
        """Get all intercept events from objectives"""
        events = []
        for obj in self.objectives:
            if isinstance(obj, InterceptObjective):
                events.extend(obj.intercept_events)
        return events
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of objective status"""
        return {
            'total': len(self.objectives),
            'completed': len(self.completed_objectives),
            'failed': len(self.failed_objectives),
            'in_progress': len(self.objectives) - len(self.completed_objectives) - len(self.failed_objectives),
            'completion_rate': len(self.completed_objectives) / len(self.objectives) if self.objectives else 0
        }