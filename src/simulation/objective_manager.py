"""
Enhanced ObjectiveManager for scenario success criteria tracking.
Manages multiple objective types and evaluates mission success.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.simulation.objectives import (
    Objective, ObjectiveType, ObjectiveStatus, ObjectiveResult,
    InterceptObjective, FuelEfficiencyObjective, TimeEfficiencyObjective,
    SurvivalObjective, NoCollisionObjective
)


class ObjectiveManager:
    """
    Manages all scenario objectives and tracks completion.
    """
    
    def __init__(self, objectives_config: Dict[str, Any]):
        """
        Initialize objective manager from configuration.
        
        Args:
            objectives_config: Dictionary with primary and secondary objectives
        """
        self.objectives: List[Objective] = []
        self.primary_objectives: List[Objective] = []
        self.secondary_objectives: List[Objective] = []
        
        # Parse primary objectives
        primary_config = objectives_config.get('primary', [])
        for obj_config in primary_config:
            obj = self._create_objective(obj_config, is_primary=True)
            if obj:
                self.objectives.append(obj)
                self.primary_objectives.append(obj)
                
        # Parse secondary objectives
        secondary_config = objectives_config.get('secondary', [])
        for obj_config in secondary_config:
            obj = self._create_objective(obj_config, is_primary=False)
            if obj:
                self.objectives.append(obj)
                self.secondary_objectives.append(obj)
                
        # Track overall status
        self.start_time = None
        self.completion_time = None
        self.all_primary_completed = False
        self.all_secondary_completed = False
        
    def _create_objective(self, config: Dict[str, Any], is_primary: bool) -> Optional[Objective]:
        """
        Create objective instance from configuration.
        
        Args:
            config: Objective configuration
            is_primary: Whether this is a primary objective
            
        Returns:
            Objective instance or None if type unknown
        """
        obj_type = config.get('type')
        
        # Handle type aliases
        if obj_type == 'fuel_remaining':
            obj_type = 'fuel_efficiency'
        elif obj_type == 'time_efficiency':
            obj_type = 'time_limit'
            
        # Create appropriate objective class
        if obj_type == 'intercept':
            return InterceptObjective(config)
        elif obj_type == 'fuel_efficiency':
            return FuelEfficiencyObjective(config)
        elif obj_type == 'time_limit':
            return TimeEfficiencyObjective(config)
        elif obj_type == 'survive':
            return SurvivalObjective(config)
        elif obj_type in ['no_collision', 'no_terrain_collision']:
            return NoCollisionObjective(config)
        elif obj_type == 'all_targets_neutralized':
            return AllTargetsNeutralizedObjective(config)
        elif obj_type == 'prioritized_intercept':
            return PrioritizedInterceptObjective(config)
        elif obj_type == 'area_denial':
            return AreaDenialObjective(config)
        else:
            print(f"Warning: Unknown objective type '{obj_type}'")
            return None
            
    def start(self, current_time: float):
        """Start tracking all objectives"""
        self.start_time = current_time
        for objective in self.objectives:
            objective.start(current_time)
            
    def update(self, asset_manager, current_time: float):
        """
        Update all objectives.
        
        Args:
            asset_manager: Asset manager instance
            current_time: Current simulation time
        """
        if self.start_time is None:
            self.start(current_time)
            
        # Evaluate each objective
        for objective in self.objectives:
            if objective.status == ObjectiveStatus.IN_PROGRESS:
                result = objective.evaluate(asset_manager, current_time)
                objective.status = result.status
                objective.progress = result.progress
                objective.message = result.message
                objective.data = result.data
                
                if result.status == ObjectiveStatus.COMPLETED:
                    objective.completion_time = current_time
                    
        # Check overall completion
        self._check_completion()
        
    def _check_completion(self):
        """Check if all primary/secondary objectives are complete"""
        # Check primary objectives
        if self.primary_objectives:
            self.all_primary_completed = all(
                obj.status == ObjectiveStatus.COMPLETED 
                for obj in self.primary_objectives
            )
        else:
            self.all_primary_completed = True
            
        # Check secondary objectives
        if self.secondary_objectives:
            self.all_secondary_completed = all(
                obj.status == ObjectiveStatus.COMPLETED
                for obj in self.secondary_objectives
            )
        else:
            self.all_secondary_completed = True
            
    def all_primary_complete(self) -> bool:
        """Check if all primary objectives are complete"""
        return self.all_primary_completed
        
    def all_complete(self) -> bool:
        """Check if all objectives are complete"""
        return self.all_primary_completed and self.all_secondary_completed
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of all objectives.
        
        Returns:
            Dictionary with objective statuses
        """
        status = {
            'primary': [],
            'secondary': [],
            'summary': {
                'total': len(self.objectives),
                'completed': 0,
                'failed': 0,
                'pending': 0,
                'in_progress': 0
            }
        }
        
        # Process primary objectives
        for obj in self.primary_objectives:
            status['primary'].append({
                'type': obj.type.value if isinstance(obj.type, ObjectiveType) else str(obj.type),
                'description': obj.description,
                'status': obj.status.value,
                'progress': obj.progress,
                'message': obj.message
            })
            
        # Process secondary objectives
        for obj in self.secondary_objectives:
            status['secondary'].append({
                'type': obj.type.value if isinstance(obj.type, ObjectiveType) else str(obj.type),
                'description': obj.description,
                'status': obj.status.value,
                'progress': obj.progress,
                'message': obj.message
            })
            
        # Count statuses
        for obj in self.objectives:
            if obj.status == ObjectiveStatus.COMPLETED:
                status['summary']['completed'] += 1
            elif obj.status == ObjectiveStatus.FAILED:
                status['summary']['failed'] += 1
            elif obj.status == ObjectiveStatus.PENDING:
                status['summary']['pending'] += 1
            elif obj.status == ObjectiveStatus.IN_PROGRESS:
                status['summary']['in_progress'] += 1
                
        return status
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of objective completion.
        
        Returns:
            Dictionary with completion summary
        """
        total = len(self.objectives)
        completed = sum(1 for obj in self.objectives 
                       if obj.status == ObjectiveStatus.COMPLETED)
        failed = sum(1 for obj in self.objectives
                    if obj.status == ObjectiveStatus.FAILED)
                    
        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'completion_rate': completed / total if total > 0 else 0,
            'all_primary_complete': self.all_primary_completed,
            'all_secondary_complete': self.all_secondary_completed
        }


# Additional objective implementations

class AllTargetsNeutralizedObjective(Objective):
    """Objective to neutralize all targets"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.initial_targets = None
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        # Get all targets
        all_assets = asset_manager.get_all_assets()
        targets = [
            asset_id for asset_id, asset_info in all_assets.items()
            if asset_info['type'] == 'target'
        ]
        
        # Track initial targets
        if self.initial_targets is None:
            self.initial_targets = len(targets)
            
        # Check if all targets neutralized
        if len(targets) == 0 and self.initial_targets > 0:
            return ObjectiveResult(
                status=ObjectiveStatus.COMPLETED,
                progress=1.0,
                message=f"All {self.initial_targets} targets neutralized"
            )
        else:
            progress = 1.0 - (len(targets) / self.initial_targets) if self.initial_targets > 0 else 0
            return ObjectiveResult(
                status=ObjectiveStatus.IN_PROGRESS,
                progress=progress,
                message=f"{len(targets)} targets remaining"
            )


class PrioritizedInterceptObjective(Objective):
    """Objective to intercept targets in priority order"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.engagement_order = config.get('engagement_order', [])
        self.current_priority_idx = 0
        self.intercepted_targets = []
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        if not self.engagement_order:
            return ObjectiveResult(
                status=ObjectiveStatus.COMPLETED,
                progress=1.0,
                message="No priority targets specified"
            )
            
        # Check current priority target
        if self.current_priority_idx >= len(self.engagement_order):
            return ObjectiveResult(
                status=ObjectiveStatus.COMPLETED,
                progress=1.0,
                message=f"All {len(self.engagement_order)} priority targets intercepted"
            )
            
        current_target = self.engagement_order[self.current_priority_idx]
        
        # Check if current target still exists
        all_assets = asset_manager.get_all_assets()
        if current_target not in all_assets:
            # Target intercepted
            self.intercepted_targets.append(current_target)
            self.current_priority_idx += 1
            
        progress = self.current_priority_idx / len(self.engagement_order)
        
        if self.current_priority_idx >= len(self.engagement_order):
            return ObjectiveResult(
                status=ObjectiveStatus.COMPLETED,
                progress=1.0,
                message=f"All priority targets intercepted in order"
            )
        else:
            return ObjectiveResult(
                status=ObjectiveStatus.IN_PROGRESS,
                progress=progress,
                message=f"Next priority target: {self.engagement_order[self.current_priority_idx]}"
            )


class AreaDenialObjective(Objective):
    """Objective to prevent targets from entering protected area"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.protected_zone = config.get('protected_zone', {})
        self.center = np.array(self.protected_zone.get('center', [25000, 25000, 0]))
        self.radius = self.protected_zone.get('radius', 5000)
        self.violations = 0
        self.max_violations = config.get('max_violations', 0)
        
    def evaluate(self, asset_manager, current_time: float) -> ObjectiveResult:
        # Check all targets
        all_assets = asset_manager.get_all_assets()
        targets_in_zone = []
        
        for asset_id, asset_info in all_assets.items():
            if asset_info['type'] == 'target':
                position = asset_info['state']['position']
                distance = np.linalg.norm(position[:2] - self.center[:2])
                
                if distance < self.radius:
                    targets_in_zone.append(asset_id)
                    
        # Update violations
        if targets_in_zone:
            self.violations = len(targets_in_zone)
            
        if self.violations > self.max_violations:
            return ObjectiveResult(
                status=ObjectiveStatus.FAILED,
                progress=0.0,
                message=f"Area breached by {self.violations} targets"
            )
        else:
            # Still successful
            threat_level = self.violations / (self.max_violations + 1)
            return ObjectiveResult(
                status=ObjectiveStatus.IN_PROGRESS,
                progress=1.0 - threat_level,
                message=f"Area secure, {self.violations}/{self.max_violations} violations"
            )